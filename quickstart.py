"""
AutoVision Quick Start Script
Launches API server and dashboard
"""
import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        🔍  AutoVision - Defect Detection System  🔍              ║
    ║                                                                   ║
    ║        Intelligent Visual Inspection for Manufacturing           ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_model_exists():
    """Check if trained model exists"""
    model_path = Path("./models/resnet18_anomaly.pth")
    
    if not model_path.exists():
        print("\n❌ Trained model not found!")
        print(f"   Expected location: {model_path.absolute()}")
        print("\n📝 Please train the model first:")
        print("   python src/train.py")
        return False
    
    print("✅ Model found:", model_path.absolute())
    return True

def check_dataset():
    """Check if dataset exists"""
    dataset_path = Path("./data/NEU-DET")
    
    if not dataset_path.exists():
        print("\n⚠️  Dataset not found at", dataset_path.absolute())
        print("   (Optional - needed only for training/testing)")
    else:
        print("✅ Dataset found:", dataset_path.absolute())
    
    return True

def start_api_server():
    """Start FastAPI server in background"""
    print("\n" + "="*70)
    print("🚀 Starting FastAPI Backend...")
    print("="*70)
    
    try:
        # Start server process
        process = subprocess.Popen(
            [sys.executable, "src/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ Waiting for server to start...")
        time.sleep(3)  # Give server time to start
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ API Server started successfully!")
            print(f"   PID: {process.pid}")
            print(f"   URL: http://localhost:8000")
            print(f"   Docs: http://localhost:8000/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start API server")
            if stderr:
                print(f"Error: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return None

def start_dashboard():
    """Start Streamlit dashboard"""
    print("\n" + "="*70)
    print("📊 Starting Streamlit Dashboard...")
    print("="*70)
    
    try:
        # Start dashboard process
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "src/dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ Waiting for dashboard to start...")
        time.sleep(5)  # Give dashboard time to start
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Dashboard started successfully!")
            print(f"   PID: {process.pid}")
            print(f"   URL: http://localhost:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start dashboard")
            if stderr:
                print(f"Error: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def show_menu():
    """Show startup menu"""
    print("\n" + "="*70)
    print("🎯 AutoVision Quick Start")
    print("="*70)
    print("\nWhat would you like to do?\n")
    print("  1. 🚀 Start API Server only")
    print("  2. 📊 Start Dashboard only")
    print("  3. 🔥 Start Both (API + Dashboard)")
    print("  4. 🧪 Test API Endpoints")
    print("  5. 📈 Run Model Evaluation")
    print("  6. 🎨 Generate Sample Inferences")
    print("  7. ℹ️  Show System Info")
    print("  8. ❌ Exit")
    print()
    
    choice = input("Enter your choice (1-8): ").strip()
    return choice

def test_api():
    """Run API tests"""
    print("\n" + "="*70)
    print("🧪 Testing API Endpoints")
    print("="*70 + "\n")
    
    if not Path("test_api.py").exists():
        print("❌ test_api.py not found")
        return
    
    os.system(f"{sys.executable} test_api.py")

def run_evaluation():
    """Run model evaluation"""
    print("\n" + "="*70)
    print("📈 Running Model Evaluation")
    print("="*70 + "\n")
    
    if not Path("test_model.py").exists():
        print("❌ test_model.py not found")
        return
    
    os.system(f"{sys.executable} test_model.py")

def generate_samples():
    """Generate sample inferences"""
    print("\n" + "="*70)
    print("🎨 Generating Sample Inferences")
    print("="*70 + "\n")
    
    print("Choose inference type:")
    print("  1. Batch inference with bounding boxes (12 samples)")
    print("  2. Single detailed inference")
    print("  3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        if Path("inference_with_bbox.py").exists():
            print("\n📦 Running batch inference...")
            os.system(f"{sys.executable} inference_with_bbox.py")
        else:
            print("❌ inference_with_bbox.py not found")
    
    if choice in ['2', '3']:
        if Path("single_inference.py").exists():
            print("\n🔍 Running single inference...")
            os.system(f"{sys.executable} single_inference.py")
        else:
            print("❌ single_inference.py not found")

def show_system_info():
    """Show system information"""
    import torch
    
    print("\n" + "="*70)
    print("ℹ️  System Information")
    print("="*70)
    
    print(f"\n📦 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"💻 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔢 CUDA Version: {torch.version.cuda}")
    
    print(f"\n📁 Current Directory: {Path.cwd()}")
    print(f"🤖 Model Path: {Path('./models/resnet18_anomaly.pth').absolute()}")
    
    # Check file sizes
    model_path = Path("./models/resnet18_anomaly.pth")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"📊 Model Size: {size_mb:.2f} MB")
    
    print("\n" + "="*70)

def main():
    """Main function"""
    print_banner()
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    print("-" * 70)
    
    if not check_model_exists():
        return
    
    check_dataset()
    
    # Main loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Start API only
            api_process = start_api_server()
            if api_process:
                print("\n✅ API is running. Press Ctrl+C to stop.")
                print(f"   Visit: http://localhost:8000/docs")
                try:
                    webbrowser.open("http://localhost:8000/docs")
                    api_process.wait()
                except KeyboardInterrupt:
                    print("\n\n⏹️  Stopping API server...")
                    api_process.terminate()
                    print("✅ Stopped")
        
        elif choice == '2':
            # Start Dashboard only
            dashboard_process = start_dashboard()
            if dashboard_process:
                print("\n✅ Dashboard is running. Press Ctrl+C to stop.")
                print(f"   Visit: http://localhost:8501")
                try:
                    webbrowser.open("http://localhost:8501")
                    dashboard_process.wait()
                except KeyboardInterrupt:
                    print("\n\n⏹️  Stopping dashboard...")
                    dashboard_process.terminate()
                    print("✅ Stopped")
        
        elif choice == '3':
            # Start both
            api_process = start_api_server()
            time.sleep(2)
            dashboard_process = start_dashboard()
            
            if api_process and dashboard_process:
                print("\n" + "="*70)
                print("✅ Both services are running!")
                print("="*70)
                print(f"\n📡 API: http://localhost:8000/docs")
                print(f"📊 Dashboard: http://localhost:8501")
                print("\nPress Ctrl+C to stop all services.")
                
                try:
                    webbrowser.open("http://localhost:8501")
                    dashboard_process.wait()
                except KeyboardInterrupt:
                    print("\n\n⏹️  Stopping services...")
                    if api_process and api_process.poll() is None:
                        api_process.terminate()
                    if dashboard_process and dashboard_process.poll() is None:
                        dashboard_process.terminate()
                    print("✅ All services stopped")
        
        elif choice == '4':
            test_api()
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            run_evaluation()
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            generate_samples()
            input("\nPress Enter to continue...")
        
        elif choice == '7':
            show_system_info()
            input("\nPress Enter to continue...")
        
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
