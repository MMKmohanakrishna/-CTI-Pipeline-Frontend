import re
from pathlib import Path
import pandas as pd

def clean_command_line(cmd):
    """Clean and normalize command line text"""
    if not isinstance(cmd, str) or cmd == '':
        return ''
    
    # Convert to lowercase
    cmd = cmd.lower()
    
    # Remove paths, keep only executable names
    cmd = re.sub(r'[c-z]:\\[^\s]*\\([^\\]+\.exe)', r'\1', cmd)
    
    # Remove specific file paths but keep general structure
    cmd = re.sub(r'\\users\\[^\\]+\\', r'\\users\\<user>\\', cmd)
    cmd = re.sub(r'\\temp\\[^\\]+', r'\\temp\\<file>', cmd)
    
    # Remove UUIDs and GUIDs
    cmd = re.sub(r'\{[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}', '<guid>', cmd)
    
    # Remove long hex strings
    cmd = re.sub(r'\b[0-9a-f]{32,}\b', '<hash>', cmd)
    
    # Normalize whitespace
    cmd = ' '.join(cmd.split())
    
    return cmd

def extract_executable_name(path):
    """Extract executable name from full path"""
    if not isinstance(path, str) or path == '':
        return ''
    
    return Path(path).name.lower()

def preprocess_event(event_dict):
    """Preprocess a single event and create combined_text"""
    # Clean command line
    cmd_cleaned = clean_command_line(event_dict.get('command_line', ''))
    
    # Extract executable names
    image_name = extract_executable_name(event_dict.get('image', ''))
    parent_name = extract_executable_name(event_dict.get('parent_image', ''))
    
    # Create combined text for model input
    combined_text = f"{image_name} {cmd_cleaned} {parent_name}".strip()
    
    return combined_text

def preprocess_dataframe(df):
    """Preprocess entire dataframe"""
    # Clean command lines
    df['command_line_cleaned'] = df['command_line'].fillna('').apply(clean_command_line)
    
    # Extract executable names
    df['image_name'] = df['image'].fillna('').apply(extract_executable_name)
    df['parent_image_name'] = df['parent_image'].fillna('').apply(extract_executable_name)
    
    # Create combined text for embedding
    df['combined_text'] = (
        df['image_name'] + ' ' + 
        df['command_line_cleaned'] + ' ' + 
        df['parent_image_name']
    )
    
    # Strip whitespace
    df['combined_text'] = df['combined_text'].str.strip()
    
    return df
