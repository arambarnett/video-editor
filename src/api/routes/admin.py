from flask import Blueprint, render_template, jsonify
import psutil
import os
import datetime
from collections import deque

admin_bp = Blueprint('admin', __name__)

# Store historical data
processing_history = deque(maxlen=100)
memory_history = deque(maxlen=100)
cpu_history = deque(maxlen=100)

@admin_bp.route('/')
def admin_dashboard():
    return render_template('admin/dashboard.html')

@admin_bp.route('/stats')
def get_stats():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Update historical data
    timestamp = datetime.datetime.now().isoformat()
    cpu_history.append({'time': timestamp, 'value': cpu_percent})
    memory_history.append({'time': timestamp, 'value': memory.percent})
    
    return jsonify({
        'system': {
            'cpu_percent': cpu_percent,
            'memory_used': memory.percent,
            'disk_used': disk.percent
        },
        'history': {
            'cpu': list(cpu_history),
            'memory': list(memory_history)
        },
        'processing': {
            'active_jobs': len(processing_history),
            'history': list(processing_history)
        }
    })
@admin_bp.route('/test', methods=['POST'])
def run_load_test():
    """Run a load test with specified parameters"""
    # Implementation for load testing
    pass
