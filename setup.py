#!/usr/bin/env python3
"""
Setup script for Real-time Audio Event Classification System
Automates the complete setup process
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

class AudioSecuritySetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data" 
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.web_dir = self.project_root / "web"
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def print_step(self, step, description):
        """Print step information"""
        print(f"\n[{step}] {description}")
        print("-" * 40)
    
    def run_command(self, command, description="", check=True):
        """Run system command with error handling"""
        if description:
            print(f"Running: {description}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=check,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                print(result.stdout)
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            return False
    
    def check_python_version(self):
        """Check Python version compatibility"""
        self.print_step("1", "Checking Python Version")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    
    def create_directories(self):
        """Create necessary project directories"""
        self.print_step("2", "Creating Project Structure")
        
        directories = [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.web_dir / "static",
            self.web_dir / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_step("3", "Installing Python Dependencies")
        
        # Check if pip is available
        if not self.run_command("pip --version", "Checking pip", check=False):
            print("❌ pip not found. Please install pip first.")
            return False
        
        # Install requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            success = self.run_command(
                f"pip install -r {requirements_file}",
                "Installing requirements.txt"
            )
            if not success:
                print("❌ Failed to install some dependencies")
                return False
        else:
            print("❌ requirements.txt not found")
            return False
        
        print("✅ Python dependencies installed successfully")
        return True
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        self.print_step("4", "Installing System Dependencies")
        
        system = platform.system().lower()
        
        if system == "linux":
            # Ubuntu/Debian dependencies
            commands = [
                "sudo apt-get update",
                "sudo apt-get install -y portaudio19-dev python3-pyaudio",
                "sudo apt-get install -y ffmpeg"
            ]
            
            for cmd in commands:
                print(f"Running: {cmd}")
                if not self.run_command(cmd, check=False):
                    print(f"Warning: Failed to run {cmd}")
        
        elif system == "darwin":  # macOS
            # Check if Homebrew is installed
            if self.run_command("brew --version", check=False):
                commands = [
                    "brew install portaudio",
                    "brew install ffmpeg"
                ]
                
                for cmd in commands:
                    self.run_command(cmd, check=False)
            else:
                print("Homebrew not found. Please install manually:")
                print("  - PortAudio: https://portaudio.com/")
                print("  - FFmpeg: https://ffmpeg.org/")
        
        elif system == "windows":
            print("Windows detected. Dependencies should install via pip.")
            print("If you encounter issues, please install:")
            print("  - Visual Studio Build Tools")
            print("  - Windows SDK")
        
        print("✅ System dependencies setup complete")
        return True
    
    def download_dataset(self):
        """Download UrbanSound8K dataset"""
        self.print_step("5", "Downloading UrbanSound8K Dataset")
        
        download_script = self.src_dir / "download_dataset.py"
        if download_script.exists():
            print("🔄 This will download ~5.6GB of data. Continue? (y/n): ", end="")
            response = input().lower().strip()
            
            if response in ['y', 'yes']:
                success = self.run_command(
                    f"cd {self.src_dir} && python download_dataset.py",
                    "Downloading dataset"
                )
                
                if success:
                    print("✅ Dataset downloaded successfully")
                    return True
                else:
                    print("❌ Dataset download failed")
                    return False
            else:
                print("⏭️  Dataset download skipped")
                return True
        else:
            print("❌ Download script not found")
            return False
    
    def test_installation(self):
        """Test the installation"""
        self.print_step("6", "Testing Installation")
        
        test_commands = [
            ("python -c 'import tensorflow as tf; print(f\"TensorFlow: {tf.__version__}\")'", "TensorFlow"),
            ("python -c 'import librosa; print(f\"Librosa: {librosa.__version__}\")'", "Librosa"),
            ("python -c 'import sounddevice as sd; print(\"SoundDevice: OK\")'", "SoundDevice"),
            ("python -c 'import flask; print(f\"Flask: {flask.__version__}\")'", "Flask"),
        ]
        
        all_passed = True
        for cmd, name in test_commands:
            if self.run_command(f"cd {self.src_dir} && {cmd}", f"Testing {name}", check=False):
                print(f"✅ {name}: OK")
            else:
                print(f"❌ {name}: Failed")
                all_passed = False
        
        return all_passed
    
    def create_demo_scripts(self):
        """Create demo and startup scripts"""
        self.print_step("7", "Creating Demo Scripts")
        
        # Desktop demo script
        desktop_demo = self.project_root / "run_desktop_demo.py"
        desktop_demo.write_text('''#!/usr/bin/env python3
"""
Quick demo script for desktop application
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from realtime_inference import main
    print("🎤 Starting Desktop Audio Monitor...")
    print("Click 'Start Monitoring' to begin real-time detection")
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
''')
        
        # Web demo script
        web_demo = self.project_root / "run_web_demo.py"
        web_demo.write_text('''#!/usr/bin/env python3
"""
Quick demo script for web application
"""
import sys
import webbrowser
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    import os
    os.chdir(Path(__file__).parent / "web")
    
    from app import socketio, app
    
    print("🌐 Starting Web Audio Monitor...")
    print("Server will start at http://localhost:5000")
    
    # Open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
''')
        
        # Training script
        train_demo = self.project_root / "run_training.py"
        train_demo.write_text('''#!/usr/bin/env python3
"""
Quick training script
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from train import AudioTrainer
    
    print("🧠 Starting Model Training...")
    print("This will take 2-3 hours on GPU, 6-8 hours on CPU")
    print("Press Ctrl+C to cancel")
    
    # Check if dataset exists
    data_dir = Path(__file__).parent / "data"
    if not (data_dir / "UrbanSound8K").exists():
        print("❌ Dataset not found!")
        print("Please run: python src/download_dataset.py")
        sys.exit(1)
    
    # Train model
    trainer = AudioTrainer(str(data_dir), 'cnn_lstm')
    history = trainer.train(epochs=80, batch_size=32)
    
    print("✅ Training complete!")
    print("Model saved to: models/best_cnn_lstm_model.keras")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
except KeyboardInterrupt:
    print("\\n⏹️  Training cancelled by user")
except Exception as e:
    print(f"❌ Error: {e}")
''')
        
        # Make scripts executable
        for script in [desktop_demo, web_demo, train_demo]:
            script.chmod(0o755)
            print(f"✅ Created: {script.name}")
        
        return True
    
    def print_final_instructions(self):
        """Print final setup instructions"""
        self.print_header("🎉 SETUP COMPLETE!")
        
        print("""
Your Real-time Audio Event Classification system is ready!

🚀 QUICK START OPTIONS:

1️⃣  Web Interface (Recommended):
   python run_web_demo.py
   → Opens browser at http://localhost:5000
   → Modern dashboard with live visualization

2️⃣  Desktop Application:
   python run_desktop_demo.py
   → Tkinter GUI for desktop use
   
3️⃣  Train Your Own Model:
   python run_training.py
   → Train on UrbanSound8K dataset
   → Takes 2-3 hours on GPU

📁 PROJECT STRUCTURE:
   src/           → Source code
   web/           → Web interface  
   data/          → Dataset (download first)
   models/        → Trained models
   logs/          → Training logs

🔧 MANUAL COMMANDS:
   # Download dataset
   cd src && python download_dataset.py
   
   # Train model
   cd src && python train.py
   
   # Optimize model
   cd src && python model_optimization.py
   
   # Run web app
   cd web && python app.py

⚠️  IMPORTANT NOTES:
   • Microphone access required for real-time detection
   • Download dataset first if training from scratch
   • GPU recommended for training (but not required)
   • Web interface works best in Chrome/Firefox

🆘 NEED HELP?
   • Check README.md for detailed documentation
   • Troubleshooting section covers common issues
   • GitHub Issues for bug reports

🛡️  SECURITY NOTICE:
   This system is for legitimate security purposes only.
   Ensure compliance with local privacy laws.

Happy monitoring! 🎤🔊
        """)
    
    def run_setup(self):
        """Run complete setup process"""
        self.print_header("🛡️ REAL-TIME AUDIO SECURITY SYSTEM SETUP")
        print("This script will set up your complete audio event classification system")
        print("Estimated time: 5-15 minutes (depending on internet speed)")
        
        steps = [
            ("Checking Python Version", self.check_python_version),
            ("Creating Directories", self.create_directories),
            ("Installing Dependencies", self.install_dependencies),
            ("System Dependencies", self.install_system_dependencies),
            ("Testing Installation", self.test_installation),
            ("Creating Demo Scripts", self.create_demo_scripts),
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    print(f"⚠️  {step_name} had issues but setup continues...")
            except Exception as e:
                failed_steps.append(step_name)
                print(f"❌ {step_name} failed: {e}")
        
        # Optional dataset download
        print("\n" + "="*60)
        print("📥 OPTIONAL: Download UrbanSound8K Dataset")
        print("="*60)
        print("You can download the dataset now or later.")
        print("Required for training, but demo works with pre-trained models.")
        
        try:
            self.download_dataset()
        except Exception as e:
            print(f"Dataset download failed: {e}")
        
        # Final results
        if failed_steps:
            print(f"\n⚠️  Setup completed with {len(failed_steps)} issues:")
            for step in failed_steps:
                print(f"   • {step}")
            print("\nYou may need to manually resolve these issues.")
        else:
            print("\n✅ Setup completed successfully!")
        
        self.print_final_instructions()


def main():
    """Main setup function"""
    setup = AudioSecuritySetup()
    setup.run_setup()


if __name__ == "__main__":
    main()