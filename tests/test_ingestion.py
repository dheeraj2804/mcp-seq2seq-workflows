import io
import zipfile
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from workflows.app import create_app
from workflows.ingestion import extract, MAX_UPLOAD_BYTES
from workflows.nlp import Normalizer


def pdf_bytes(text=None, encrypted=False, padding=0):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    if padding:
        writer.add_metadata({'/Subject': 'x' * padding})
    if encrypted:
        writer.encrypt('secret')
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def docx_bytes():
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>The cobaltbridge rollout begins Monday.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>The owner is Alex.</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
    return output.getvalue()


@pytest.fixture
def client(tmp_path):
    return create_app(tmp_path / 'runs.sqlite', Normalizer(tmp_path/'missing.pt')).test_client()


def upload(client, data, name):
    return client.post('/api/documents', data={'file': (io.BytesIO(data), name)}, content_type='multipart/form-data')


def test_pdf_upload_and_search(client):
    response = upload(client, pdf_bytes('The zephyrcove review is scheduled for Friday morning.'), 'Launch.pdf')
    assert response.status_code == 201, response.get_json()
    payload = response.get_json()
    assert payload['metadata']['pages'] == 1 and payload['metadata']['format'] == 'pdf'
    result = client.post('/api/workflows', json={'query': 'zephyrcove'}).get_json()
    assert 'Friday' in result['answer'] and '[Page 1]' in result['sources'][0]['text']
    assert result['sources'][0]['metadata']['filename'] == 'Launch.pdf'


def test_docx_body_and_tables(client):
    response = upload(client, docx_bytes(), 'Checklist.DOCX')
    assert response.status_code == 201, response.get_json()
    result = client.post('/api/workflows', json={'query': 'cobaltbridge'}).get_json()
    assert 'Monday' in result['answer'] and 'Alex' in result['answer']


def test_large_text_and_metadata(client):
    data = ('Reference notes about equipment. ' * 10000 + 'uniqueharbor shipment arrives Tuesday.').encode()
    assert len(data) > 100000
    response = upload(client, data, 'large-notes.txt')
    assert response.status_code == 201
    assert response.get_json()['characters'] == len(data)
    listing = client.get('/api/documents').get_json()
    assert listing['limits']['upload_bytes'] == MAX_UPLOAD_BYTES
    assert listing['documents'][-1]['metadata']['bytes'] == len(data)
    result = client.post('/api/workflows', json={'query': 'uniqueharbor'}).get_json()
    assert 'Tuesday' in result['answer']


@pytest.mark.parametrize('data,name,expected', [
    (b'', 'empty.txt', 'empty'),
    (b'fake', 'fake.pdf', 'valid PDF'),
    (b'fake', 'fake.docx', 'Unable to read'),
    (b'not a supported document', 'notes.doc', 'Legacy .doc'),
    (b'\xff\x80'*20, 'bad.txt', 'encoding'),
    (b'abc\x00def'*10, 'binary.txt', 'binary'),
])
def test_rejected_uploads(client, data, name, expected):
    response = upload(client, data, name)
    assert response.status_code == 400
    assert expected in response.get_json()['error']
    assert len(client.get('/api/documents').get_json()['documents']) == 7


def test_scanned_and_encrypted_pdf(client):
    for data, error in [(pdf_bytes(), 'OCR'), (pdf_bytes('Private notes in an encrypted file.', True), 'Password-protected')]:
        response = upload(client, data, 'notes.pdf')
        assert response.status_code == 400
        assert error in response.get_json()['error']


def test_limits(client):
    response = upload(client, b'x' * (MAX_UPLOAD_BYTES + 1), 'huge.txt')
    assert response.status_code == 400 and '20 MB' in response.get_json()['error']
    with pytest.raises(ValueError, match='2 million'):
        extract(b'x' * 2000001, '.txt')
    writer = PdfWriter()
    for _ in range(501):
        writer.add_blank_page(width=100, height=100)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match='500 pages'):
        extract(output.getvalue(), '.pdf')


def test_pdf_over_old_file_limit(client):
    data = pdf_bytes('The silvermeadow delivery arrives on Wednesday.', padding=200000)
    assert len(data) > 100000
    response = upload(client, data, 'larger-report.pdf')
    assert response.status_code == 201, response.get_json()
    result = client.post('/api/workflows', json={'query': 'silvermeadow'}).get_json()
    assert 'Wednesday' in result['answer']
