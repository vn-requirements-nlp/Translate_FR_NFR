# Step 01: Tạo môi trường ảo (virtual environment) để cô lập thư viện
python -m venv .venv

# Step 02: Kích hoạt môi trường ảo (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Step 03: Cài dependencies cho repo
pip install -r requirements.txt

# Step 04: Translate (chỉnh batch_size tương ứng với số câu muốn dịch 1 phiên)
python translate_requirements.py `
  --in_txt Dataset_Full_EN.txt `
  --out_txt Dataset_Full_Vietnamese.txt `
  --model gpt-5-mini `
  --batch_size 150

# Step 05: Resume khi bị ngưng
python translate_requirements.py --in_txt Dataset_Full_EN.txt --out_txt Dataset_Full_VI.txt --resume
