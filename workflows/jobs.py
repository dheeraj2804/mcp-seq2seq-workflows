"""Bounded in-process SSE jobs with explicit asynchronous cancellation."""
import asyncio
import json
import threading
import time
import uuid
from collections import deque
from flask import Response,jsonify,request,stream_with_context
from .progress import EventTrace,progress_scope

class Job:
    def __init__(self):
        self.id=uuid.uuid4().hex; self.created=time.time(); self.done=False; self.cancelled=False
        self.events=deque(maxlen=500); self.sequence=0; self.condition=threading.Condition()
        self.loop=None;self.task=None
    def emit(self,event):
        with self.condition:
            self.sequence+=1; self.events.append((self.sequence,event));self.condition.notify_all()
    def bind(self,loop,task):
        self.loop=loop;self.task=task
        if self.cancelled: task.cancel()
    def cancel(self):
        if self.done:return False
        self.cancelled=True
        if self.loop and self.task:
            try:self.loop.call_soon_threadsafe(self.task.cancel)
            except RuntimeError:pass
        return True


def register_jobs(app,runner):
    jobs={};lock=threading.Lock()
    def worker(job,body):
        try:
            with app.app_context(),progress_scope(job.emit):
                result,status=runner(body,EventTrace(),job)
            job.emit({'type':'result','data':result})
        except Exception:
            job.emit({'type':'error','data':{'error':'The workflow could not start. Check the question, provider and document selections.'}})
        finally:
            with job.condition:
                job.done=True;job.emit({'type':'done','data':{}})
    @app.post('/api/workflow-jobs')
    def start_job():
        body=request.get_json(silent=True)
        if not isinstance(body,dict) or not isinstance(body.get('query'),str) or not 0<len(body['query'].strip())<=1000:
            raise ValueError('Enter a question of at most 1000 characters')
        with lock:
            for key in list(jobs):
                if jobs[key].done and (time.time()-jobs[key].created>900 or len(jobs)>20):del jobs[key]
            if sum(not job.done for job in jobs.values())>=2:
                return jsonify(error='Two workflows are already running. Wait or cancel one.'),429
            job=Job();jobs[job.id]=job
        threading.Thread(target=worker,args=(job,body.copy()),daemon=True).start()
        return jsonify(job_id=job.id),202
    @app.delete('/api/workflow-jobs/<job_id>')
    def cancel_job(job_id):
        job=jobs.get(job_id)
        if not job:return jsonify(error='Workflow job not found'),404
        return jsonify(status='cancelling' if job.cancel() else 'finished')
    @app.get('/api/workflow-jobs/<job_id>/events')
    def events(job_id):
        job=jobs.get(job_id)
        if not job:return jsonify(error='Workflow job not found'),404
        try:after=int(request.headers.get('Last-Event-ID','0'))
        except ValueError:after=0
        def generate():
            nonlocal after
            while True:
                with job.condition:
                    pending=[(number,event) for number,event in job.events if number>after]
                    if not pending and not job.done:
                        job.condition.wait(timeout=10)
                        pending=[(number,event) for number,event in job.events if number>after]
                    finished=job.done
                for number,event in pending:
                    after=number
                    yield f'id: {number}\ndata: {json.dumps(event)}\n\n'
                if finished:return
                if not pending:yield ': heartbeat\n\n'
        return Response(stream_with_context(generate()),mimetype='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})
    return jobs
