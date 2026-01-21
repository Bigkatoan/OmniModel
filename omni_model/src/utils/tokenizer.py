# src/utils/tokenizer.py

from tokenizers import Tokenizer
import torch

class BPE_Tokenizer:
    def __init__(self, tokenizer_path, max_seq_len=32):
        """
        Wrapper class cho HuggingFace Tokenizer (BPE).
        Args:
            tokenizer_path (str): Đường dẫn file .json tokenizer đã train.
            max_seq_len (int): Độ dài cố định của chuỗi token.
        """
        self.max_seq_len = max_seq_len
        
        try:
            # Load tokenizer từ file json
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            
            # Cấu hình Padding & Truncation tự động
            # Padding: Thêm [PAD] id vào cuối câu cho đủ độ dài
            self.tokenizer.enable_padding(
                direction='right',
                pad_id=self.tokenizer.token_to_id("[PAD]"),
                pad_token="[PAD]",
                length=max_seq_len
            )
            
            # Truncation: Cắt bớt nếu câu quá dài
            self.tokenizer.enable_truncation(
                max_length=max_seq_len,
                direction='right'
            )
            
        except Exception as e:
            print(f"LỖI CRITICAL: Không load được tokenizer tại {tokenizer_path}")
            print("Hãy chắc chắn bạn đã chạy script train tokenizer trước!")
            raise e

    def encode(self, text):
        """
        Chuyển text thành Tensor ID.
        Output shape: (max_seq_len,)
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Encode (đã bao gồm padding/truncation)
        encoding = self.tokenizer.encode(text)
        
        # Chuyển sang PyTorch Tensor
        ids = torch.tensor(encoding.ids, dtype=torch.long)
        
        # Tạo Attention Mask (1 là token thật, 0 là padding)
        mask = torch.tensor(encoding.attention_mask, dtype=torch.long)
        
        return ids, mask

    def decode(self, ids):
        """
        Chuyển Tensor ID ngược lại thành Text.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        # skip_special_tokens=True để loại bỏ [PAD], [BOS], [EOS]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

# --- Đoạn test nhanh (chỉ chạy khi gọi trực tiếp file này) ---
if __name__ == "__main__":
    # Test giả lập
    # Lưu ý: Cần có file data/tokenizer.json thật mới chạy được
    try:
        tok = BPE_Tokenizer("../../data/tokenizer.json", max_seq_len=10)
        text = "Hello robot, find the red cup"
        ids, mask = tok.encode(text)
        
        print(f"Input: {text}")
        print(f"IDs: {ids}")
        print(f"Mask: {mask}")
        print(f"Decoded: {tok.decode(ids)}")
    except:
        print("Chưa tìm thấy file tokenizer.json để test.")