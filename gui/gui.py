import ctypes
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import numpy as np
import os
import sys

last_name_raw = None
last_tc_raw = None
last_diag_text = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'model.pth')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'model', 'scaler.pkl')
ICON_PATH = os.path.join(BASE_DIR, '..', 'assets', 'icon.ico')

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translations = {
    "en": {

        "select_file": "Select Patient File",
        "drop_here": "Drop patient.txt file here",
        "no_file": "No file selected yet",
        "developed_by": "Developed by Ahmet Can Çömez",
        "warning": "Model file 'model.pth' not found.\nPlease train the model first (train.py).",
        "prediction": "Prediction: ",
        "label_name": "Name",
        "label_tc": "ID Number",
        "label_diagnosis": "Diagnosis",
        "malignant": "Malignant",
        "benign": "Benign",
        "confidence": "Confidence",
        "not_found": "Not found"
    },
    "tr": {
        "select_file": "Hasta Dosyası Seç",
        "drop_here": "Hasta.txt dosyasını buraya bırakın",
        "no_file": "Henüz bir dosya seçilmedi",
        "developed_by": "Geliştiren: Ahmet Can Çömez",
        "warning": "Model dosyası bulunamadı.\nLütfen önce modeli eğitin (train.py).",
        "prediction": "Teşhis: ",
        "label_name": "Ad Soyad",
        "label_tc": "TC Kimlik No",
        "label_diagnosis": "Teşhis",
        "malignant": "Kötü Huylu",
        "benign": "İyi Huylu",
        "confidence": "Güven",
        "not_found": "Bulunamadı"
    }
}
current_lang = "en"


def switch_language(lang_code):
    global current_lang
    current_lang = lang_code
    btn.config(text=translations[current_lang]["select_file"])
    drop_area.config(text=translations[current_lang]["drop_here"])
    developer_label.config(text=translations[current_lang]["developed_by"])
    lang_btn.config(text="EN" if current_lang == "tr" else "TR")

    if not hasattr(root, "name_label"):
        result_label.config(text=translations[current_lang]["no_file"])
    else:
        name_text = root.name_label.cget("text")
        tc_text = root.tc_label.cget("text")
        diagnosis_text = root.diagnosis_label.cget("text")

        name_value = name_text.split(": ", 1)[-1]
        tc_value = tc_text.split(": ", 1)[-1]

        import re
        match = re.search(r"\(.*?%[\d.]+\)", diagnosis_text)
        if match:
            confidence_raw = match.group(0)
            confidence_value = re.search(r"%[\d.]+", confidence_raw).group(0)
            diagnosis_key = translations[current_lang][
                "benign"] if "Benign" in diagnosis_text or "İyi" in diagnosis_text else translations[current_lang][
                "malignant"]
            confidence_label = translations[current_lang]["confidence"]
            diagnosis_line = f"{translations[current_lang]['label_diagnosis']}: {diagnosis_key} ({confidence_label}: {confidence_value})"
        else:
            diagnosis_line = f"{translations[current_lang]['label_diagnosis']}: ?"

        root.name_label.config(text=f"{translations[current_lang]['label_name']}: {name_value}")
        root.tc_label.config(text=f"{translations[current_lang]['label_tc']}: {tc_value}")
        root.diagnosis_label.config(text=diagnosis_line)

    if hasattr(root, "name_label") and last_diag_text is not None:
        name = last_name_raw if last_name_raw else translations[current_lang]["not_found"]
        tc = last_tc_raw if last_tc_raw else translations[current_lang]["not_found"]

        if tc != translations[current_lang]["not_found"]:
            tc = tc[:3] + '*' * (len(tc) - 5) + tc[-2] + tc[-1]

        root.name_label.config(text=f"{translations[current_lang]['label_name']}: {name}")
        root.tc_label.config(text=f"{translations[current_lang]['label_tc']}: {tc}")
        root.diagnosis_label.config(text=f"{translations[current_lang]['label_diagnosis']}: {last_diag_text}")


class CancerNet(nn.Module):
    def __init__(self):
        super(CancerNet, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if not os.path.exists(MODEL_PATH):
    root = tk.Tk()
    root.title("Breast Cancer Diagnostics App")
    root.geometry("500x300")
    root.resizable(False, False)
    root.iconbitmap(ICON_PATH)
    tk.Label(root, text=translations[current_lang]["warning"], font=("Segoe UI", 12, "bold"), fg="red",
             wraplength=400, justify="center").pack(expand=True)
    tk.Label(root, text=translations[current_lang]["developed_by"], font=("Segoe UI", 8, "italic"), fg="#888888").pack(
        pady=(10, 10))
    root.mainloop()
    sys.exit()

model = CancerNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def predict_patient_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip().split(',')

        features_raw = line[-30:]
        try:
            features = np.array([float(x) for x in features_raw], dtype=np.float32)
        except:
            return "Invalid format. Last 30 values must be numeric health features.", "black", 0, None, None, None

        metadata = line[:-30]
        name_raw = None
        tc_raw = None

        if len(metadata) == 2:
            name_raw, tc_raw = metadata
        elif len(metadata) == 1:
            if metadata[0].strip().isdigit():
                tc_raw = metadata[0]
            else:
                name_raw = metadata[0]

        import joblib
        scaler = joblib.load(SCALER_PATH)
        features = scaler.transform([features])[0]

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        class_names = [translations[current_lang]["malignant"], translations[current_lang]["benign"]]
        diagnosis_line = f"{class_names[pred]} (%{confidence * 100:.2f})"

        return None, pred, name_raw, tc_raw, diagnosis_line

    except Exception as e:
        return f"Hata: {str(e)}", -1, None, None, None



def handle_file(file_path):
    global last_name_raw, last_tc_raw, last_diag_text

    result_text, diagnosis, name_raw, tc_raw, diagnosis_line = predict_patient_file(file_path)

    last_name_raw = name_raw
    last_tc_raw = tc_raw
    last_diag_text = diagnosis_line

    for attr in ["name_label", "tc_label", "diagnosis_label"]:
        if hasattr(root, attr):
            getattr(root, attr).destroy()
            delattr(root, attr)

    result_label.config(text="")

    if diagnosis in [0, 1]:
        diagnosis_color = "#B22222" if diagnosis == 0 else "#2E8B57"

        name = name_raw if name_raw else translations[current_lang]["not_found"]
        tc = tc_raw if tc_raw else translations[current_lang]["not_found"]

        if tc != translations[current_lang]["not_found"]:
            tc = tc[:3] + '*' * (len(tc) - 5) + tc[-2] + tc[-1]

        root.name_label = tk.Label(frame, text=f"{translations[current_lang]['label_name']}: {name}",
                                   font=("Segoe UI", 14, "bold"), justify="center")
        root.name_label.pack(pady=(5, 1))

        root.tc_label = tk.Label(frame, text=f"{translations[current_lang]['label_tc']}: {tc}",
                                 font=("Segoe UI", 14, "bold"), justify="center")
        root.tc_label.pack(pady=(1, 1))

        root.diagnosis_label = tk.Label(frame, text=f"{translations[current_lang]['label_diagnosis']}: {diagnosis_line}",
                                        font=("Segoe UI", 14, "bold"),
                                        fg=diagnosis_color, justify="center")
        root.diagnosis_label.pack(pady=(1, 5))
    else:
        result_label.config(text=result_text, fg="black")



def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        handle_file(file_path)


def drop(event):
    file_path = event.data.strip('{}')
    if os.path.isfile(file_path):
        handle_file(file_path)


try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    root = TkinterDnD.Tk()
except ImportError:
    root = tk.Tk()

root.title("Breast Cancer Diagnostics App")
root.geometry("800x700")
root.resizable(False, False)
root.iconbitmap(ICON_PATH)
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)


def toggle_language():
    global current_lang
    current_lang = "tr" if current_lang == "en" else "en"
    switch_language(current_lang)
    lang_btn.config(text="EN" if current_lang == "tr" else "TR")


lang_btn = tk.Button(root, text="TR", command=toggle_language,
                     font=("Segoe UI", 10, "bold"),
                     width=4, height=1, relief="ridge", bd=2)
lang_btn.place(x=720, y=10)

btn = tk.Button(frame, text=translations[current_lang]["select_file"], command=open_file,
                font=("Segoe UI", 12, "bold"))
btn.pack(pady=20)

drop_area = tk.Label(
    frame,
    text=translations[current_lang]["drop_here"],
    relief="groove",
    borderwidth=2,
    bg="white",
    width=40,
    height=8,
    font=("Segoe UI", 10, "italic")
)
drop_area.pack(pady=(1, 0))

try:
    drop_area.drop_target_register(DND_FILES)
    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind('<<Drop>>', drop)
except:
    pass

result_label = tk.Label(frame, text=translations[current_lang]["no_file"], font=("Segoe UI", 14, "bold"),
                        justify="left")
result_label.pack(pady=(10, 5))

developer_label = tk.Label(root, text=translations[current_lang]["developed_by"],
                           font=("Segoe UI", 9, "italic"), fg="#888888")
developer_label.pack(side="bottom", pady=20)

root.mainloop()


def main():
    root.mainloop()
