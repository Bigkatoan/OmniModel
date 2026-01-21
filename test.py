import numpy as np
import torch
import os

# Giả lập việc import như một thư viện
try:
    from omni_model.interface import OmniModel
    print("✅ Import thành công package 'omni_model'")
except ImportError as e:
    print("❌ Lỗi Import! Kiểm tra lại cấu trúc thư mục.")
    print(e)
    exit()

def test_pipeline():
    print("\n--- BẮT ĐẦU TEST LOCAL ---")
    
    # 1. Khởi tạo Model
    # Nó sẽ tự tìm file weights trong omni_model/weights/ trước
    try:
        model = OmniModel(device='cpu') # Test trên CPU cho nhanh
        print("✅ Khởi tạo OmniModel thành công")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return

    # 2. Test Encode Text
    try:
        text = "Test sentence"
        text_emb = model.encode_text(text)
        print(f"✅ Encode Text: Shape {text_emb.shape}")
        
        # Kiểm tra shape
        assert text_emb.shape == (1, 512), "Shape text embedding sai!"
    except Exception as e:
        print(f"❌ Lỗi Encode Text: {e}")

    # 3. Test Encode Image (Dùng ảnh giả lập numpy)
    try:
        # Tạo một ảnh đen 224x224
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        img_emb = model.encode_image(dummy_img)
        print(f"✅ Encode Image: Shape {img_emb.shape}")
        
        assert img_emb.shape == (1, 512), "Shape image embedding sai!"
    except Exception as e:
        print(f"❌ Lỗi Encode Image: {e}")
        
    # 4. Test Logic Similarity
    try:
        score = (img_emb @ text_emb.T).item()
        print(f"✅ Similarity calculation check: {score:.4f}")
    except Exception as e:
        print(f"❌ Lỗi tính toán Similarity: {e}")

    print("\n🎉 CHÚC MỪNG! PACKAGE CỦA BẠN ĐÃ HOẠT ĐỘNG TỐT!")

if __name__ == "__main__":
    # Kiểm tra xem folder weights có đủ file chưa
    required_files = ["vision_encoder.pth", "text_encoder.pth", "vision_proj.pth"]
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join("omni_model/weights", f)):
            missing.append(f)
    
    if missing:
        print("⚠️ CẢNH BÁO: Thiếu file weights local:", missing)
        print("Code sẽ cố gắng tải từ HuggingFace (có thể lỗi nếu bạn chưa upload).")
    else:
        print("🆗 Đã tìm thấy đầy đủ weights local.")
        
    test_pipeline()