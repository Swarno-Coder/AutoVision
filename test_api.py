"""
API Test Script
Test all AutoVision API endpoints
"""
import requests
import os
import base64
from PIL import Image
from io import BytesIO

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE = "./data/NEU-DET/validation/images/crazing/crazing_241.jpg"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ API is healthy")
            print(f"   Status: {result.get('status')}")
            print(f"   Model loaded: {result.get('model_loaded')}")
            print(f"   Grad-CAM loaded: {result.get('gradcam_loaded')}")
            print(f"   Device: {result.get('device')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        print("   Make sure the API server is running: python src/app.py")
        return False

def test_root():
    """Test root endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Root Endpoint")
    print("="*80)
    
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            result = response.json()
            print("✅ Root endpoint working")
            print(f"   Service: {result.get('service')}")
            print(f"   Version: {result.get('version')}")
            print(f"   Classes: {', '.join(result.get('classes', []))}")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_info():
    """Test model info endpoint"""
    print("\n" + "="*80)
    print("TEST 3: Model Info")
    print("="*80)
    
    try:
        response = requests.get(f"{API_URL}/info")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model info retrieved")
            print(f"   Model type: {result.get('model_type')}")
            print(f"   Number of classes: {result.get('num_classes')}")
            print(f"   Input size: {result.get('input_size')}")
            return True
        else:
            print(f"❌ Info endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict():
    """Test basic prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 4: Basic Prediction")
    print("="*80)
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Image: {os.path.basename(TEST_IMAGE)}")
            print(f"   Prediction: {result.get('prediction').upper()}")
            print(f"   Confidence: {result.get('confidence', 0)*100:.2f}%")
            print(f"   Class ID: {result.get('class_id')}")
            
            # Show top 3 predictions
            print("\n   Top 3 Predictions:")
            for i, pred in enumerate(result.get('top_3_predictions', [])[:3], 1):
                print(f"      {i}. {pred['class']}: {pred['probability']*100:.2f}%")
            
            return True
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_gradcam():
    """Test prediction with Grad-CAM endpoint"""
    print("\n" + "="*80)
    print("TEST 5: Prediction with Grad-CAM")
    print("="*80)
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict/gradcam", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Grad-CAM prediction successful")
            print(f"   Prediction: {result.get('prediction').upper()}")
            print(f"   Confidence: {result.get('confidence', 0)*100:.2f}%")
            print(f"   Explanation: {result.get('explanation', 'N/A')[:100]}...")
            
            # Check if Grad-CAM image is included
            if 'gradcam_image' in result:
                print("   ✓ Grad-CAM heatmap generated")
                
                # Optionally save the heatmap
                save_gradcam = input("\n   Save Grad-CAM image? (y/n): ").lower()
                if save_gradcam == 'y':
                    img_data = base64.b64decode(result['gradcam_image'])
                    img = Image.open(BytesIO(img_data))
                    output_path = './api_test_gradcam.jpg'
                    img.save(output_path)
                    print(f"   💾 Saved to: {output_path}")
            
            return True
        else:
            print(f"❌ Grad-CAM prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_predict():
    """Test batch prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 6: Batch Prediction")
    print("="*80)
    
    # Find multiple test images
    test_dir = "./data/NEU-DET/validation/images/crazing"
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    try:
        # Get 3 test images
        image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:3]
        
        if len(image_files) < 2:
            print("❌ Not enough test images found")
            return False
        
        # Prepare files
        files = []
        for img_file in image_files:
            img_path = os.path.join(test_dir, img_file)
            files.append(('files', open(img_path, 'rb')))
        
        # Make request
        response = requests.post(f"{API_URL}/batch/predict", files=files)
        
        # Close files
        for _, file_obj in files:
            file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful")
            print(f"   Total images: {result.get('total_images')}")
            print("\n   Results:")
            
            for item in result.get('results', []):
                if 'error' not in item:
                    print(f"      {item['filename']}: {item['prediction']} ({item['confidence']*100:.1f}%)")
                else:
                    print(f"      {item['filename']}: ERROR - {item['error']}")
            
            return True
        else:
            print(f"❌ Batch prediction failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*80)
    print("🧪 AutoVision API Test Suite")
    print("="*80)
    print(f"API URL: {API_URL}")
    print(f"Test Image: {TEST_IMAGE}")
    print("="*80)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Model Info", test_info),
        ("Basic Prediction", test_predict),
        ("Grad-CAM Prediction", test_predict_gradcam),
        ("Batch Prediction", test_batch_predict),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
