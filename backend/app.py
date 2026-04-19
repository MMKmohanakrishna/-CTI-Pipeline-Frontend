from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import io
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from preprocess import preprocess_event, preprocess_dataframe

app = FastAPI(title="CTI Pipeline - Malware Detection API")

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
minilm_model = None
minilm_tokenizer = None
device = None

class EventInput(BaseModel):
    event_id: str
    image: str
    parent_image: str
    command_line: str
    timestamp: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    event_details: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    total_events: int
    malicious_count: int
    benign_count: int
    threat_count: int = 0
    results: List[Dict[str, Any]]

class FilePathInput(BaseModel):
    file_path: str

class EventTextInput(BaseModel):
    event_text: str

@app.on_event("startup")
async def load_models():
    """Load MiniLM model at startup"""
    global minilm_model, minilm_tokenizer, device
    
    print("Loading MiniLM model...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model path - use retrained model if available, fallback to original
    retrained_model_dir = Path(__file__).parent.parent / "models" / "minilm_retrained"
    original_model_dir = Path(__file__).parent.parent / "models" / "minilm"
    
    if retrained_model_dir.exists():
        model_dir = retrained_model_dir
        print(f"✓ Using RETRAINED model (balanced 50/50 dataset)")
    else:
        model_dir = original_model_dir
        print(f"⚠ Using ORIGINAL model (100% malicious training data - expect high false positives)")
    
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")
    
    try:
        minilm_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        minilm_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        minilm_model.to(device)
        minilm_model.eval()
        print("✓ MiniLM model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load MiniLM model: {e}")

def predict_with_minilm(text: str, threshold: float = 0.75, image_path: str = '', command_line: str = '', 
                        event_data: dict = None) -> tuple:
    """Hybrid prediction: rule-based + ML model with comprehensive security rules
    
    Args:
        text: Combined text for ML model
        threshold: Confidence threshold
        image_path: Full path to the executable
        command_line: Command line arguments
        event_data: Extended event data including network fields (protocol, dest_ip, dest_port, etc.)
    """
    if not text.strip():
        return 0, [0.9, 0.1]
    
    text_lower = text.lower()
    image_lower = image_path.lower()
    cmd_lower = command_line.lower()
    import re
    
    # Extract network event data if available
    if event_data is None:
        event_data = {}
    
    protocol = event_data.get('protocol', '').lower()
    initiated = event_data.get('initiated', '').lower()
    source_ip = event_data.get('source_ip', '').lower()
    dest_ip = event_data.get('dest_ip', '').lower()
    dest_port = event_data.get('dest_port', '').lower()
    source_port = event_data.get('source_port', '').lower()
    user = event_data.get('user', '').lower()
    
    # Helper functions for IP classification
    def is_private_ip(ip: str) -> bool:
        """Check if IP is private (RFC1918), link-local, or multicast"""
        if not ip:
            return False
        return bool(re.match(r'^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|169\.254\.|224\.0\.0\.|127\.0\.0\.1|fe80:|ff[0-9a-f]{2}:)', ip))
    
    def is_mdns_multicast(ip: str) -> bool:
        """Check if IP is mDNS multicast address"""
        return ip in ['224.0.0.251', 'ff02::fb']
    
    def is_trusted_system_path(path: str) -> bool:
        """Check if path is in trusted system folders"""
        path_lower = path.lower()
        trusted = ['c:\\windows\\system32\\', 'c:\\windows\\syswow64\\', 
                   'c:\\program files\\', 'c:\\program files (x86)\\',
                   'c:/windows/system32/', 'c:/windows/syswow64/',
                   'c:/program files/', 'c:/program files (x86)/']
        return any(path_lower.startswith(t) for t in trusted)
    
    def is_multicast_broadcast(ip: str) -> bool:
        """Check if IP is multicast or broadcast"""
        if not ip:
            return False
        return bool(re.match(r'^(224\.|239\.|255\.255\.255\.255|ff[0-9a-f]{2}:)', ip))
    
    def is_public_ip(ip: str) -> bool:
        """Check if IP is public (not private, not multicast, not loopback)"""
        if not ip or ip == '-':
            return False
        return not is_private_ip(ip) and not is_multicast_broadcast(ip)
    
    # ===== BENIGN RULES (CHECK FIRST - SHORT CIRCUIT) =====
    
    # Benign Rule 1: mDNS multicast / link-local system traffic
    if protocol == 'udp' and dest_port == '5353':
        if is_mdns_multicast(dest_ip) or is_mdns_multicast(source_ip):
            if is_trusted_system_path(image_path):
                return 0, [0.99, 0.01]  # mDNS system traffic
    
    # Benign Rule 2: Trusted system process to private network on known service ports
    if is_trusted_system_path(image_path):
        known_ports = ['53', '5353', '137', '138', '139', '445', '80', '443', '123', '67', '68']
        if dest_port in known_ports and is_private_ip(dest_ip):
            return 0, [0.98, 0.02]  # Trusted system to private IP on known port
    
    # Benign Rule 3: Established local multicast or broadcast from trusted system
    if is_multicast_broadcast(dest_ip) and is_trusted_system_path(image_path):
        return 0, [0.97, 0.03]  # System multicast/broadcast
    
    # Benign Rule 4: Non-initiated inbound/listening from system process to local/private
    if initiated == 'false' and is_trusted_system_path(image_path):
        if is_private_ip(dest_ip) or not dest_ip:
            return 0, [0.96, 0.04]  # Inbound to system process
    
    # Benign Rule 5: Legitimate svchost.exe from system32 with no suspicious patterns
    is_svchost = 'svchost.exe' in text_lower
    is_system32_svchost = is_trusted_system_path(image_path) and 'svchost.exe' in image_lower
    
    if is_svchost and is_system32_svchost:
        if not any(bad in text_lower for bad in ['powershell -enc', 'iex', 'downloadstring', 'invoke-']):
            # Network event with private/multicast IP or minimal command line
            if is_private_ip(dest_ip) or is_multicast_broadcast(dest_ip) or not cmd_lower or len(cmd_lower.strip()) < 10:
                return 0, [0.98, 0.02]  # Legitimate svchost network activity
    
    # Benign Rule 6: Common legitimate cmd.exe commands (standalone or piped to benign commands)
    is_cmd = 'cmd.exe' in image_lower and is_trusted_system_path(image_path)
    if is_cmd and cmd_lower:
        # Check if NOT chained with suspicious patterns
        has_suspicious = any(sus in cmd_lower for sus in ['powershell -enc', 'invoke-', 'iex', 'downloadstring',
                                                           'net localgroup administrators', 'mimikatz', 'procdump',
                                                           'certutil -decode', 'bitsadmin', 'regsvr32'])
        has_chaining = any(chain in cmd_lower for chain in ['&&', ';'])
        
        if not has_suspicious and not has_chaining:
            # Common benign single commands
            benign_cmd_patterns = [
                r'^cmd\.exe /c dir\b', r'^cmd\.exe /c echo\b', r'^cmd\.exe /c cd\b',
                r'^cmd\.exe /c type\b', r'^cmd\.exe /c ver\b', r'^cmd\.exe /c time\b',
                r'^cmd\.exe /c date\b', r'^cmd\.exe /c cls\b', r'^cmd\.exe /c tasklist\b',
                r'^cmd\.exe /c help\b', r'^cmd\.exe /c copy\b', r'^cmd\.exe /c move\b',
                r'^cmd\.exe /c del\b', r'^cmd\.exe /c mkdir\b', r'^cmd\.exe /c rmdir\b',
                r'^cmd\.exe /c ping\b', r'^cmd\.exe /c whoami\s*$', r'^cmd\.exe /c net user\s*$'
            ]
            for pattern in benign_cmd_patterns:
                if re.match(pattern, cmd_lower):
                    return 0, [0.95, 0.05]  # Common benign cmd command
            
            # ipconfig or systeminfo piped to benign commands (findstr, more, sort, find)
            benign_pipes = ['| findstr', '| more', '| sort', '| find ']
            has_benign_pipe = any(pipe in cmd_lower for pipe in benign_pipes)
            
            # Standalone or piped to benign commands
            if re.match(r'^cmd\.exe /c ipconfig\b', cmd_lower):
                return 0, [0.94, 0.06]  # ipconfig (standalone or piped to benign)
            if re.match(r'^cmd\.exe /c systeminfo\b', cmd_lower):
                if has_benign_pipe or '&&' not in cmd_lower:
                    return 0, [0.94, 0.06]  # systeminfo piped to benign or standalone
    
    # ===== MALICIOUS RULES (CHECK AFTER BENIGN) =====
    
    # Rule 1: Executable in TEMP / AppData
    if re.search(r'\\(appdata|temp|tmp)\\.*\.exe', text_lower):
        return 1, [0.02, 0.98]
    
    # Rule 2: Fake system filenames (typosquatting)
    fake_system_names = ['svch0st', 'scvhost', 'expl0rer', 'svhost', 'cxplorer', 'iexpl0re', 
                         'taskmg r', 'csrss ', 'lsasss', 'smss ', 'winl0gon']
    for fake in fake_system_names:
        if fake in text_lower:
            return 1, [0.02, 0.98]
    
    # Rule 3: PowerShell with encoded commands
    powershell_malicious = [
        'powershell -enc', 'powershell -e ', 'powershell.exe -enc',
        '-encodedcommand', 'iex ', 'invoke-expression', 'downloadstring',
        '-nop -noni -w hidden', '-executionpolicy bypass -enc',
        '-executionpolicy bypass -e ', '-executionpolicy bypass iex',
    ]
    for pattern in powershell_malicious:
        if pattern in text_lower:
            return 1, [0.03, 0.97]
    
    # Rule 4: Commands that download files from internet
    download_patterns = [
        'wget ', 'curl ', 'invoke-webrequest', 'iwr ', 'downloadfile',
        'certutil -urlcache', 'certutil.exe -urlcache', 'bitsadmin /transfer',
        'webclient', 'net.webclient', 'start-bitstransfer'
    ]
    for pattern in download_patterns:
        if pattern in text_lower:
            return 1, [0.03, 0.97]
    
    # Rule 5: Outbound connection to public IP on suspicious ports
    suspicious_ports = ['4444', '8080', '1337', '3389', '5555', '6666', '7777']
    for port in suspicious_ports:
        if f':{port}' in text_lower or f' {port}' in text_lower:
            # Check if not connecting to localhost/private IPs
            if not re.search(r'(127\.0\.0\.1|localhost|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)', text_lower):
                return 1, [0.05, 0.95]
    
    # Rule 6: Long Base64-like strings (>80 continuous alphanumeric+/)
    if re.search(r'[A-Za-z0-9+/]{80,}', text):
        return 1, [0.04, 0.96]
    
    # Rule 7: Scripts launched from Temp/Downloads
    script_extensions = ['.ps1', '.vbs', '.js', '.bat', '.cmd', '.hta']
    for ext in script_extensions:
        if ext in text_lower and re.search(r'\\(temp|tmp|downloads|appdata)\\', text_lower):
            return 1, [0.03, 0.97]
    
    # Rule 8: Masqueraded or double extensions
    double_extensions = ['.pdf.exe', '.jpg.exe', '.png.exe', '.doc.exe', '.xls.exe',
                         '.txt.exe', '.zip.exe', '.jpg.scr', '.pdf.scr', '.rar.exe']
    for ext in double_extensions:
        if ext in text_lower:
            return 1, [0.02, 0.98]
    
    # Additional malicious patterns - context-aware reconnaissance detection
    # These commands are only malicious when chained with && or ; or | with other recon/malicious commands
    recon_commands = ['whoami', 'net user', 'net localgroup', 'ipconfig /all', 'systeminfo']
    recon_found = sum(1 for cmd in recon_commands if cmd in text_lower)
    
    # If multiple recon commands are chained (&&, ;, |), it's malicious
    if recon_found >= 2 or (recon_found >= 1 and any(chain in text_lower for chain in ['&&', ' ; ', ' | '])):
        # Check if it's actually chaining malicious activity
        if any(mal in text_lower for mal in ['powershell -enc', 'invoke-', 'iex', 'downloadstring', 
                                              'net localgroup administrators', 'mimikatz', 'procdump']):
            return 1, [0.03, 0.97]
    
    # Clearly malicious patterns (not context-dependent)
    always_malicious = [
        'wmic process', 'reg add', 'schtasks /create', 'rundll32.exe javascript', 
        'rundll32 javascript', 'regsvr32 /s /u', 'certutil -decode',
        'sc create', 'servicepointmanager',
        'invoke-mimikatz', 'mimikatz', 'procdump', 'psexec', 
        'net localgroup administrators', '/add'
    ]
    for pattern in always_malicious:
        if pattern in text_lower:
            return 1, [0.04, 0.96]
    
    # Detect standalone powershell.exe with suspicious context (not covered by Rule 3)
    if 'powershell.exe' in text_lower:
        # Check if it has malicious indicators
        suspicious_ps_patterns = ['-nop', '-w hidden', '-windowstyle hidden', 
                                  'bypass', 'invoke-', 'iex', 'downloadstring',
                                  'webclient', 'net.', 'bitstransfer']
        for sus_pattern in suspicious_ps_patterns:
            if sus_pattern in text_lower:
                return 1, [0.03, 0.97]
    
    # ===== NETWORK-BASED MALICIOUS RULES =====
    
    # Network Rule 1: External public IP / unusual outbound port
    if initiated == 'true' and is_public_ip(dest_ip):
        # Outbound to public IP is suspicious
        suspicious_network_ports = ['4444', '8080', '1337', '9001', '5555', '444', '6666', '7777', '31337']
        if dest_port in suspicious_network_ports:
            return 1, [0.02, 0.98]  # Outbound to public IP on suspicious port
        # Even standard outbound can be suspicious from non-system paths
        if not is_trusted_system_path(image_path):
            return 1, [0.10, 0.90]  # Non-system process to public IP
    
    # Network Rule 2: Process in AppData/Temp with network activity
    if re.search(r'\\(appdata|temp|tmp)\\', image_lower):
        if dest_ip and (is_public_ip(dest_ip) or dest_port):
            return 1, [0.02, 0.98]  # AppData/Temp process with network activity
    
    # Network Rule 3: Unusual protocol/port from system binary to public IP
    if is_trusted_system_path(image_path) and is_public_ip(dest_ip):
        try:
            port_num = int(dest_port) if dest_port else 0
            # System process to public IP on high port (> 1024, not common)
            if port_num > 1024 and port_num not in [8080, 8443]:
                # Could be suspicious unless it's a browser/updater
                if 'svchost' in image_lower and port_num > 10000:
                    return 1, [0.15, 0.85]  # svchost to public IP on unusual high port
        except:
            pass
    
    # Network Rule 4: Encoded/obfuscated command with network activity
    if any(pattern in text_lower for pattern in ['-enc', 'iex', 'invoke-webrequest', 'downloadstring', 
                                                   'curl', 'wget', 'certutil']):
        if dest_ip or protocol:  # Has network activity
            return 1, [0.02, 0.98]  # Obfuscated command with network
    
    # ===== THRESHOLD RULE: Same Process Name in Image Path =====
    # Detect when process name appears multiple times in the full path (potential process masquerading)
    if image_path:
        # Extract process name from path
        process_name = image_path.split('\\')[-1].lower() if '\\' in image_path else image_path.split('/')[-1].lower()
        process_name_base = process_name.replace('.exe', '').replace('.dll', '').replace('.scr', '')
        
        # Count occurrences of process name in the full path (excluding the final executable name)
        path_without_exe = '\\'.join(image_path.split('\\')[:-1]).lower() if '\\' in image_path else '/'.join(image_path.split('/')[:-1]).lower()
        
        # Check if process name appears in the path multiple times (threshold rule)
        if process_name_base and len(process_name_base) > 3:  # Only check meaningful names
            occurrence_count = path_without_exe.count(process_name_base)
            if occurrence_count >= 1:  # Process name appears in path before the executable
                # This could be legitimate (e.g., C:\Program Files\Chrome\chrome.exe)
                # But suspicious if in temp/appdata
                if any(suspicious in path_without_exe for suspicious in ['temp', 'appdata', 'downloads', 'public']):
                    return 1, [0.05, 0.95]  # Threshold: Same process name in suspicious path
    
    # Rule 1: File located in trusted system folders
    trusted_paths = [
        'c:\\windows\\system32\\', 'c:\\windows\\syswow64\\',
        'c:\\program files\\', 'c:\\program files (x86)\\'
    ]
    # Check if image path is in trusted location AND command is clean
    for path in trusted_paths:
        if path in text_lower and not any(bad in text_lower for bad in ['powershell -enc', 'iex', 'downloadstring']):
            # Additional check: verify it's a known good process
            if any(proc in text_lower for proc in ['svchost.exe', 'lsass.exe', 'csrss.exe', 'winlogon.exe']):
                return 0, [0.98, 0.02]
    
    # Rule 2: Common legitimate process names with clean commands
    legitimate_processes = [
        'notepad.exe', 'calc.exe', 'mspaint.exe', 'taskmgr.exe', 'regedit.exe',
        'explorer.exe explorer.exe', 'chrome.exe https', 'chrome.exe http',
        'firefox.exe https', 'firefox.exe http', 'msedge.exe https',
        'winword.exe', 'excel.exe', 'powerpnt.exe', 'outlook.exe',
        '7zfm.exe', 'winrar.exe', 'vlc.exe'
    ]
    for proc in legitimate_processes:
        if proc in text_lower:
            return 0, [0.96, 0.04]
    
    # Rule 3: Clean command-line (simple operations)
    clean_commands = [
        'cmd.exe /c dir', 'cmd.exe /c ping', 'cmd.exe /c ipconfig',
        'cmd.exe /c type', 'cmd.exe /c ver', 'cmd.exe /c hostname',
        'get-date', 'get-location', 'get-childitem', 'start-process calc',
        'msiexec.exe /i', 'msiexec /i', 'netstat -ano', 'get-process',
        'netstat.exe -ano'
    ]
    for cmd in clean_commands:
        if cmd in text_lower:
            return 0, [0.95, 0.05]
    
    # Rule 4: Network connection to private IPs
    private_ip_patterns = [
        r'192\.168\.\d{1,3}\.\d{1,3}',
        r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        r'172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}',
        r'127\.0\.0\.1',
        r'localhost'
    ]
    for pattern in private_ip_patterns:
        if re.search(pattern, text_lower):
            return 0, [0.93, 0.07]
    
    # Fall back to ML model for uncertain cases
    try:
        inputs = minilm_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = minilm_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # Apply higher threshold for ML predictions (conservative)
            if probs[0][1] >= 0.85:  # 85% threshold for ML-only
                pred = 1
            else:
                pred = 0
        
        return pred, probs[0].cpu().tolist()
    except:
        # If ML fails, default to benign
        return 0, [0.7, 0.3]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CTI Pipeline - Malware Detection API",
        "version": "1.0.0",
        "model": "MiniLM-L12-H384",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict/single",
            "predict_batch": "/predict/batch"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": minilm_model is not None,
        "device": str(device)
    }

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(event: EventInput):
    """Predict single event"""
    try:
        # Preprocess event
        combined_text = preprocess_event({
            'event_id': event.event_id,
            'image': event.image,
            'parent_image': event.parent_image,
            'command_line': event.command_line,
            'timestamp': event.timestamp
        })
        
        # Predict
        pred, probs = predict_with_minilm(combined_text, image_path=event.image, command_line=event.command_line)
        
        # Format response
        prediction_label = "malicious" if pred == 1 else "benign"
        confidence = probs[pred]
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "benign": round(probs[0], 4),
                "malicious": round(probs[1], 4)
            },
            "event_details": {
                "event_id": event.event_id,
                "image": event.image,
                "command_line": event.command_line,
                "combined_text": combined_text
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Predict batch of events from CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Read CSV - handle inconsistent fields
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')
        
        # Validate required columns
        required_cols = ['event_id', 'image', 'parent_image', 'command_line', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns.tolist())}"
            )
        
        # Keep only required columns to avoid issues with extra columns
        df = df[required_cols]
        
        # Fill NaN values
        df = df.fillna('')
        
        # Preprocess
        df = preprocess_dataframe(df)
        
        # Separate events by rule-based detection (fast) vs ML-needed (slow)
        ml_needed_indices = []
        results = []
        malicious_count = 0
        benign_count = 0
        threat_count = 0
        
        # Track malicious detections per process for threshold rule
        process_malicious_counts = {}
        
        # First pass: Rule-based pattern matching (instant) - use same function
        for idx, row in df.iterrows():
            combined_text = row['combined_text']
            process_image = str(row['image']).lower()
            
            # Use the same comprehensive rule-based detection with original data
            pred, probs = predict_with_minilm(combined_text, image_path=row['image'], command_line=row['command_line'])
            
            # Track malicious detections per process
            if pred == 1:
                process_malicious_counts[process_image] = process_malicious_counts.get(process_image, 0) + 1
            
            # Check if it was rule-based (high confidence) or needs ML
            if probs[pred] >= 0.93:  # Rule-based detection (high confidence)
                # Store temporarily - we'll apply threshold logic after counting all
                results.append({
                    "row_index": int(idx),
                    "event_id": str(row['event_id']),
                    "image": str(row['image']),
                    "command_line": str(row['command_line']),
                    "prediction_raw": pred,  # Store raw prediction
                    "confidence": round(probs[pred], 4),
                    "probabilities": {
                        "benign": round(probs[0], 4),
                        "malicious": round(probs[1], 4)
                    }
                })
            else:
                # Need ML inference for uncertain cases
                ml_needed_indices.append((idx, row, combined_text))
        
        # Second pass: Batch ML inference for uncertain cases (faster than one-by-one)
        if ml_needed_indices:
            texts = [item[2] for item in ml_needed_indices]
            
            # Batch tokenization (much faster)
            inputs = minilm_tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Batch inference
            with torch.no_grad():
                outputs = minilm_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Process batch results
            for i, (idx, row, _) in enumerate(ml_needed_indices):
                pred = 1 if probs[i][1] >= 0.85 else 0
                process_image = str(row['image']).lower()
                
                # Track malicious detections per process
                if pred == 1:
                    process_malicious_counts[process_image] = process_malicious_counts.get(process_image, 0) + 1
                
                results.append({
                    "row_index": int(idx),
                    "event_id": str(row['event_id']),
                    "image": str(row['image']),
                    "command_line": str(row['command_line']),
                    "prediction_raw": pred,  # Store raw prediction
                    "confidence": round(float(probs[i][pred]), 4),
                    "probabilities": {
                        "benign": round(float(probs[i][0]), 4),
                        "malicious": round(float(probs[i][1]), 4)
                    }
                })
        
        # Count unique processes with malicious detections
        processes_with_malicious = {proc: count for proc, count in process_malicious_counts.items() if count > 0}
        
        # Apply threshold rule ONLY if there are multiple events from the SAME process
        # If processes are different, follow normal malicious/benign rules
        for result in results:
            process_image = result['image'].lower()
            pred = result['prediction_raw']
            mal_count_for_process = process_malicious_counts.get(process_image, 0)
            
            # Count how many events exist for this specific process
            events_for_this_process = sum(1 for r in results if r['image'].lower() == process_image)
            
            # Only apply threshold rule if this process appears multiple times
            if events_for_this_process >= 2:
                # Threshold rule: If >= 2 malicious detections for same process, ALL events show as malicious
                if mal_count_for_process >= 2:
                    result['prediction'] = "malicious"
                    malicious_count += 1
                elif mal_count_for_process == 1 and pred == 1:
                    # Only 1 malicious detection for this process, and this is that malicious one
                    result['prediction'] = "threat"
                    threat_count += 1
                elif mal_count_for_process == 1 and pred == 0:
                    # Only 1 malicious detection for this process, but this event is benign
                    result['prediction'] = "benign"
                    benign_count += 1
                else:
                    # No malicious detections for this process
                    result['prediction'] = "benign"
                    benign_count += 1
            else:
                # Single event for this process - follow normal malicious/benign rule
                if pred == 1:
                    result['prediction'] = "malicious"
                    malicious_count += 1
                else:
                    result['prediction'] = "benign"
                    benign_count += 1
            
            # Remove temporary field
            del result['prediction_raw']
        
        # Sort results by row_index to maintain original order
        results.sort(key=lambda x: x['row_index'])
        
        return {
            "total_events": len(df),
            "malicious_count": malicious_count,
            "benign_count": benign_count,
            "threat_count": threat_count,
            "results": results
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

def analyze_file_by_type(file_path: Path) -> tuple:
    """Analyze file based on extension and path using comprehensive security rules"""
    import re
    
    file_name = file_path.name.lower()
    file_ext = file_path.suffix.lower()
    file_location = str(file_path).lower().replace('\\', '/')
    
    is_malicious = False
    reasons = []
    confidence = 0.95  # High confidence for rule-based detection
    
    # ===== BENIGN DATAPATH RULES (Check first for whitelisting) =====
    
    # Rule 1: System32 (Trusted Windows System Folder)
    if any(trusted in file_location for trusted in ['c:/windows/system32/', 'c:/windows/syswow64/', 'c:/windows/winsxs/']):
        return 0, [0.98, 0.02], ["File in trusted Windows system folder"]
    
    # Rule 2: Program Files (Trusted Application Area)
    if any(trusted in file_location for trusted in ['c:/program files/', 'c:/program files (x86)/', 'd:/apps/company/']):
        return 0, [0.97, 0.03], ["File in trusted application folder (Program Files)"]
    
    # Rule 3: User Document/Media Folders (non-executable)
    doc_media_paths = ['/documents/', '/pictures/', '/desktop/', '/downloads/', '/music/', '/videos/']
    non_exec_exts = ['.txt', '.pdf', '.docx', '.xlsx', '.pptx', '.png', '.jpg', '.jpeg', '.gif', '.mp3', '.mp4', '.avi']
    if any(path in file_location for path in doc_media_paths) and file_ext in non_exec_exts:
        return 0, [0.96, 0.04], [f"Document/media file ({file_ext}) in user folder"]
    
    # Rule 4: System Tools in Correct Location
    known_system_tools = ['explorer.exe', 'notepad.exe', 'calc.exe', 'mspaint.exe', 'taskmgr.exe', 
                          'cmd.exe', 'regedit.exe', 'svchost.exe', 'lsass.exe', 'csrss.exe']
    if file_name in known_system_tools:
        if 'c:/windows/system32/' in file_location or 'c:/windows/syswow64/' in file_location:
            return 0, [0.99, 0.01], [f"Legitimate system tool ({file_name}) in correct location"]
    
    # ===== MALICIOUS DATAPATH RULES (High-confidence threats) =====
    
    # Rule 5: Executables in TEMP or AppData (Very Common Malware Location)
    malware_locations = ['/appdata/local/temp/', '/appdata/roaming/', '/temp/', '/tmp/']
    exec_exts = ['.exe', '.dll', '.ps1', '.js', '.vbs', '.bat', '.scr', '.cmd']
    if any(loc in file_location for loc in malware_locations) and file_ext in exec_exts:
        is_malicious = True
        reasons.append(f"Executable ({file_ext}) in high-risk location (Temp/AppData)")
        confidence = 0.98
    
    # Rule 6: Fake System File Names (Impersonation / Typos)
    fake_system_names = ['svch0st.exe', 'scvhost.exe', 'expl0rer.exe', 'winupd.exe', 
                        'micr0soft.exe', 'iexpl0re.exe', 'svhost.exe', 'csrss.exe', 
                        'lsasss.exe', 'smss.exe', 'winl0gon.exe']
    for fake in fake_system_names:
        if fake in file_name:
            is_malicious = True
            reasons.append(f"Fake system filename detected: {fake}")
            confidence = 0.99
            break
    
    # Rule 7: Double-Extension Masquerade (Classic Malware Trick)
    double_exts = ['.pdf.exe', '.jpg.scr', '.png.exe', '.doc.exe', '.docx.exe', 
                   '.xls.exe', '.xlsx.exe', '.txt.exe', '.zip.exe', '.rar.exe',
                   '.gif.exe', '.mp3.exe', '.avi.exe', '.jpg.exe', '.pdf.scr']
    for dext in double_exts:
        if dext in file_name:
            is_malicious = True
            reasons.append(f"Double extension masquerade detected: {dext}")
            confidence = 0.97
            break
    
    # Rule 8: Scripts Executed from User Folders
    script_exts = ['.ps1', '.js', '.vbs', '.bat', '.cmd', '.hta', '.wsf']
    user_script_paths = ['/users/', '/documents/', '/downloads/', '/desktop/', '/appdata/']
    if file_ext in script_exts and any(path in file_location for path in user_script_paths):
        is_malicious = True
        reasons.append(f"Script file ({file_ext}) in user directory")
        confidence = 0.94
    
    # Rule 9: Unknown EXE in User Directories
    user_dirs = ['/desktop/', '/downloads/', '/documents/']
    if file_ext == '.exe' and any(udir in file_location for udir in user_dirs):
        # Exception for .msi installers
        if file_ext != '.msi':
            is_malicious = True
            reasons.append("Executable (.exe) in user directory")
            confidence = 0.92
    
    # Rule 10: Executable in Startup Folder (Persistence Trick)
    startup_path = '/appdata/roaming/microsoft/windows/start menu/programs/startup/'
    if startup_path in file_location and file_ext in exec_exts:
        is_malicious = True
        reasons.append("Executable in Startup folder (persistence mechanism)")
        confidence = 0.96
    
    # Rule 11: Drivers (.sys) Outside System32\drivers
    if file_ext == '.sys':
        if 'system32/drivers' not in file_location and 'syswow64/drivers' not in file_location:
            is_malicious = True
            reasons.append("Driver file (.sys) outside system drivers folder")
            confidence = 0.95
    
    # Rule 12: Executable in Root of C:\ (Rare and Suspicious)
    if re.match(r'^c:/[^/]+\.(exe|dll|bat|ps1|vbs|js)$', file_location):
        is_malicious = True
        reasons.append("Executable in root of C:\\ drive (highly suspicious)")
        confidence = 0.97
    
    # Additional checks: Suspicious keywords in filename
    suspicious_keywords = ['crack', 'keygen', 'patch', 'loader', 'injector', 'bypass', 
                          'hack', 'exploit', 'payload', 'backdoor', 'trojan', 'rat',
                          'malware', 'ransomware', 'miner', 'stealer']
    for keyword in suspicious_keywords:
        if keyword in file_name:
            is_malicious = True
            reasons.append(f"Suspicious keyword in filename: {keyword}")
            confidence = 0.93
            break
    
    # System files in wrong location
    system_processes = ['csrss.exe', 'lsass.exe', 'svchost.exe', 'winlogon.exe', 'smss.exe', 'services.exe']
    if file_name in system_processes:
        if 'system32' not in file_location and 'syswow64' not in file_location:
            is_malicious = True
            reasons.append(f"System process ({file_name}) in non-system location")
            confidence = 0.98
    
    # Final decision
    if is_malicious:
        return 1, [1 - confidence, confidence], reasons
    else:
        # Default to benign with lower confidence if no rules matched
        if not reasons:
            reasons.append("No malicious indicators detected")
        return 0, [0.85, 0.15], reasons

@app.post("/predict/batch-path", response_model=BatchPredictionResponse)
async def predict_batch_from_path(file_input: FilePathInput):
    """Analyze any file path - can be CSV file or direct file analysis"""
    try:
        file_path = file_input.file_path.strip()
        
        # Convert to Path object and normalize path
        path = Path(file_path)
        
        # If relative path, try from project root
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / file_path
        
        # Check if file exists - if not, analyze the path string anyway
        file_exists = False
        if not path.exists():
            # Try current working directory
            alt_path = Path.cwd() / file_path
            if alt_path.exists():
                path = alt_path
                file_exists = True
            else:
                # File doesn't exist - analyze path string only
                file_exists = False
        else:
            file_exists = True
        
        # Check if it's a CSV file (event log analysis) or other file type (file analysis)
        if path.suffix.lower() == '.csv' and file_exists:
            # Original CSV processing logic - Read CSV
            df = pd.read_csv(path, on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')
            
            # Validate required columns
            required_cols = ['event_id', 'image', 'parent_image', 'command_line', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns.tolist())}"
                )
            
            # Keep only required columns
            df = df[required_cols]
            
            # Fill NaN values
            df = df.fillna('')
            
            # Preprocess
            df = preprocess_dataframe(df)
            
            # Same processing logic as batch upload
            ml_needed_indices = []
            results = []
            malicious_count = 0
            benign_count = 0
            threat_count = 0
            
            # Track malicious detections per process for threshold rule
            process_malicious_counts = {}
            
            # First pass: Rule-based pattern matching
            for idx, row in df.iterrows():
                combined_text = row['combined_text']
                process_image = str(row['image']).lower()
                
                # Use comprehensive rule-based detection
                pred, probs = predict_with_minilm(combined_text, image_path=row['image'], command_line=row['command_line'])
                
                # Track malicious detections per process
                if pred == 1:
                    process_malicious_counts[process_image] = process_malicious_counts.get(process_image, 0) + 1
                
                # Check if it was rule-based (high confidence) or needs ML
                if probs[pred] >= 0.93:
                    results.append({
                        "row_index": int(idx),
                        "event_id": str(row['event_id']),
                        "image": str(row['image']),
                        "command_line": str(row['command_line']),
                        "prediction_raw": pred,
                        "confidence": round(probs[pred], 4),
                        "probabilities": {
                            "benign": round(probs[0], 4),
                            "malicious": round(probs[1], 4)
                        }
                    })
                else:
                    ml_needed_indices.append((idx, row, combined_text))
            
            # Second pass: Batch ML inference
            if ml_needed_indices:
                texts = [item[2] for item in ml_needed_indices]
                
                inputs = minilm_tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = minilm_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                for i, (idx, row, _) in enumerate(ml_needed_indices):
                    pred = 1 if probs[i][1] >= 0.85 else 0
                    process_image = str(row['image']).lower()
                    
                    # Track malicious detections per process
                    if pred == 1:
                        process_malicious_counts[process_image] = process_malicious_counts.get(process_image, 0) + 1
                    
                    results.append({
                        "row_index": int(idx),
                        "event_id": str(row['event_id']),
                        "image": str(row['image']),
                        "command_line": str(row['command_line']),
                        "prediction_raw": pred,
                        "confidence": round(float(probs[i][pred]), 4),
                        "probabilities": {
                            "benign": round(float(probs[i][0]), 4),
                            "malicious": round(float(probs[i][1]), 4)
                        }
                    })
            
            # Apply threshold rule ONLY if same process appears multiple times
            for result in results:
                process_image = result['image'].lower()
                pred = result['prediction_raw']
                mal_count_for_process = process_malicious_counts.get(process_image, 0)
                
                # Count how many events exist for this specific process
                events_for_this_process = sum(1 for r in results if r['image'].lower() == process_image)
                
                # Only apply threshold rule if this process appears multiple times
                if events_for_this_process >= 2:
                    # Threshold rule: If >= 2 malicious detections for same process, ALL events show as malicious
                    if mal_count_for_process >= 2:
                        result['prediction'] = "malicious"
                        malicious_count += 1
                    elif mal_count_for_process == 1 and pred == 1:
                        result['prediction'] = "threat"
                        threat_count += 1
                    elif mal_count_for_process == 1 and pred == 0:
                        result['prediction'] = "benign"
                        benign_count += 1
                    else:
                        result['prediction'] = "benign"
                        benign_count += 1
                else:
                    # Single event for this process - follow normal malicious/benign rule
                    if pred == 1:
                        result['prediction'] = "malicious"
                        malicious_count += 1
                    else:
                        result['prediction'] = "benign"
                        benign_count += 1
                
                del result['prediction_raw']
            
            # Sort results by row_index
            results.sort(key=lambda x: x['row_index'])
            
            return {
                "total_events": len(df),
                "malicious_count": malicious_count,
                "benign_count": benign_count,
                "threat_count": threat_count,
                "results": results
            }
        
        else:
            # Direct file analysis based on file type and location
            pred, probs, reasons = analyze_file_by_type(path)
            
            prediction_label = "malicious" if pred == 1 else "benign"
            malicious_count = 1 if pred == 1 else 0
            benign_count = 1 if pred == 0 else 0
            
            # Create analysis text from reasons
            analysis_text = "; ".join(reasons) if reasons else "Standard file analysis"
            
            result = {
                "row_index": 0,
                "event_id": "FILE_ANALYSIS",
                "image": str(path),
                "command_line": f"File Type: {path.suffix.upper()} | {analysis_text}",
                "prediction": prediction_label,
                "confidence": round(probs[pred], 4),
                "probabilities": {
                    "benign": round(probs[0], 4),
                    "malicious": round(probs[1], 4)
                }
            }
            
            return {
                "total_events": 1,
                "malicious_count": malicious_count,
                "benign_count": benign_count,
                "results": [result]
            }
    
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction from path failed: {str(e)}")

@app.post("/predict/event-text", response_model=BatchPredictionResponse)
async def predict_event_from_text(event_input: EventTextInput):
    """Predict single event from CSV-formatted text"""
    try:
        event_text = event_input.event_text.strip()
        
        # Parse CSV text - split by comma but handle quoted values
        import csv
        from io import StringIO
        
        # Try to parse as CSV
        csv_reader = csv.reader(StringIO(event_text))
        rows = list(csv_reader)
        
        if len(rows) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Get the row data
        row = rows[0]
        
        # Try to detect format
        import json
        
        # Format 1: Standard format (event_id, image, parent_image, command_line, timestamp, ...)
        # Format 2: Sysmon extended format (event_id, timestamp, processId, image, ..., json_blob)
        
        # Check if there's a JSON blob in the row (usually contains network details)
        json_data = None
        for i, cell in enumerate(row):
            if cell.strip().startswith('{') and cell.strip().endswith('}'):
                try:
                    json_data = json.loads(cell)
                    break
                except:
                    pass
        
        # If we have JSON data, use that format (Sysmon extended)
        if json_data:
            # Sysmon format: event_id, timestamp, processId, image, ..., json_blob
            event_dict = {
                'event_id': row[0] if len(row) > 0 else '',
                'timestamp': row[1] if len(row) > 1 else '',
                'image': json_data.get('Image', row[3] if len(row) > 3 else ''),
                'parent_image': '',  # Not in this format
                'command_line': '',  # Not in this format
                'protocol': json_data.get('Protocol', '').lower(),
                'initiated': json_data.get('Initiated', '').lower(),
                'source_ip': json_data.get('SourceIp', ''),
                'source_hostname': json_data.get('SourceHostname', ''),
                'source_port': json_data.get('SourcePort', ''),
                'dest_ip': json_data.get('DestinationIp', ''),
                'dest_hostname': json_data.get('DestinationHostname', ''),
                'dest_port': json_data.get('DestinationPort', ''),
                'user': json_data.get('User', '')
            }
        else:
            # Standard format
            if len(row) < 5:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Expected at least 5 columns (event_id, image, parent_image, command_line, timestamp), got {len(row)}"
                )
            
            # Create event dictionary with base fields
            event_dict = {
                'event_id': row[0],
                'image': row[1],
                'parent_image': row[2],
                'command_line': row[3],
                'timestamp': row[4]
            }
            
            # Parse extended network event fields (if available)
            if len(row) > 5:
                try:
                    idx = 5
                    if len(row) > idx: event_dict['protocol'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['initiated'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['source_ip'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['source_hostname'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['source_port'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['source_port_name'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['dest_ip'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['dest_hostname'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['dest_port'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['dest_port_name'] = row[idx]; idx += 1
                    if len(row) > idx: event_dict['user'] = row[idx]; idx += 1
                except:
                    pass  # If parsing extended fields fails, continue with base fields
        
        # Preprocess event
        combined_text = preprocess_event(event_dict)
        
        # Predict using comprehensive rules with network event data
        pred, probs = predict_with_minilm(
            combined_text, 
            image_path=event_dict['image'], 
            command_line=event_dict['command_line'],
            event_data=event_dict
        )
        
        prediction_label = "malicious" if pred == 1 else "benign"
        malicious_count = 1 if pred == 1 else 0
        benign_count = 1 if pred == 0 else 0
        
        result = {
            "row_index": 0,
            "event_id": str(event_dict['event_id']),
            "image": str(event_dict['image']),
            "command_line": str(event_dict['command_line']),
            "prediction": prediction_label,
            "confidence": round(probs[pred], 4),
            "probabilities": {
                "benign": round(probs[0], 4),
                "malicious": round(probs[1], 4)
            }
        }
        
        return {
            "total_events": 1,
            "malicious_count": malicious_count,
            "benign_count": benign_count,
            "results": [result]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event text prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
