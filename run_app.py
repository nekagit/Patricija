#!/usr/bin/env python3
"""
Starter Script für XAI Kreditprüfung
Startet Frontend und Backend in separaten Terminals
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_backend():
    """Startet das Backend in einem neuen Terminal"""
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend-Verzeichnis nicht gefunden!")
        return False
    
    os.chdir(backend_dir)
    
    # Starte Backend direkt
    print("🚀 Starte Backend...")
    subprocess.run([sys.executable, "main.py"])

def run_frontend():
    """Startet das Frontend in einem neuen Terminal"""
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend-Verzeichnis nicht gefunden!")
        return False
    
    os.chdir(frontend_dir)
    
    # Starte Frontend direkt
    print("🚀 Starte Frontend...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port", "8501"])

def main():
    """Hauptfunktion"""
    print("🎯 XAI Kreditprüfung - Starter")
    print("=" * 40)
    
    # Prüfe Betriebssystem
    if os.name == 'nt':  # Windows
        import subprocess
        import threading
        
        def start_backend():
            subprocess.run(["start", "cmd", "/k", "python", "run_app.py", "--backend"], shell=True)
        
        def start_frontend():
            subprocess.run(["start", "cmd", "/k", "python", "run_app.py", "--frontend"], shell=True)
        
        # Starte beide in separaten Terminals
        print("🔄 Starte Backend und Frontend in separaten Terminals...")
        
        backend_thread = threading.Thread(target=start_backend)
        frontend_thread = threading.Thread(target=start_frontend)
        
        backend_thread.start()
        time.sleep(2)  # Kurze Pause
        frontend_thread.start()
        
        print("✅ Beide Services gestartet!")
        print("📱 Frontend: http://localhost:8501")
        print("🔧 Backend: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
    else:  # Linux/Mac
        print("🔄 Starte Backend und Frontend...")
        
        # Starte Backend im Hintergrund
        backend_process = subprocess.Popen([
            sys.executable, "run_app.py", "--backend"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)  # Warte bis Backend gestartet ist
        
        # Starte Frontend
        frontend_process = subprocess.Popen([
            sys.executable, "run_app.py", "--frontend"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Beide Services gestartet!")
        print("📱 Frontend: http://localhost:8501")
        print("🔧 Backend: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
        try:
            # Warte auf Beendigung
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Beende Services...")
            backend_process.terminate()
            frontend_process.terminate()

if __name__ == "__main__":
    # Prüfe Kommandozeilenargumente
    if len(sys.argv) > 1:
        if sys.argv[1] == "--backend":
            run_backend()
        elif sys.argv[1] == "--frontend":
            run_frontend()
        else:
            print("❌ Unbekanntes Argument. Verwende --backend oder --frontend")
    else:
        main()
