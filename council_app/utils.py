"""
Utility functions for the council app.
"""
import os
import psutil


def get_system_info():
    """
    Detect hardware constraints and return info dict.
    
    Returns a dictionary with:
    - ram_gb: Total RAM in gigabytes
    - cpu_count: Number of CPU cores
    - is_constrained: True if system has limited resources
    - warning: Warning message if constrained, None otherwise
    - recommendations: List of recommendations for constrained systems
    """
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    cpu_count = os.cpu_count() or 1
    
    info = {
        'ram_gb': round(ram_gb, 1),
        'cpu_count': cpu_count,
        'is_constrained': False,
        'warning': None,
        'recommendations': []
    }
    
    # Check for memory constraints
    if ram_gb < 16:
        info['is_constrained'] = True
        
        if ram_gb < 8:
            info['warning'] = (
                f"Running on {info['ram_gb']}GB RAM. "
                "LLM responses may be very slow (5+ minutes). "
                "Use only the smallest models."
            )
            info['recommendations'] = [
                'Use tinyllama or qwen2.5:0.5b for faster responses',
                'Select only 2-3 models at a time',
                'Expect response times of 3-10 minutes',
            ]
        else:
            info['warning'] = (
                f"Running on {info['ram_gb']}GB RAM. "
                "LLM responses may take several minutes. "
                "Consider using smaller models for faster results."
            )
            info['recommendations'] = [
                'Smaller models (phi3:mini, tinyllama) respond faster',
                'Larger models (llama3:8b) may take 5+ minutes',
                'Response times vary by model complexity',
            ]
    
    return info


def format_duration(seconds):
    """
    Format a duration in seconds to a human-readable string.
    
    Examples:
        format_duration(45) -> "45s"
        format_duration(125) -> "2m 5s"
        format_duration(3665) -> "1h 1m 5s"
    """
    if seconds is None:
        return "—"
    
    seconds = int(seconds)
    
    if seconds < 60:
        return f"{seconds}s"
    
    minutes, secs = divmod(seconds, 60)
    
    if minutes < 60:
        if secs:
            return f"{minutes}m {secs}s"
        return f"{minutes}m"
    
    hours, mins = divmod(minutes, 60)
    parts = [f"{hours}h"]
    if mins:
        parts.append(f"{mins}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts)
