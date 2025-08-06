# 🫁 Lung Cancer Detection using CNN

A deep learning-based application for detecting lung cancer from CT scan images using **Convolutional Neural Networks (CNN)**.  
The system is integrated with a **Streamlit** web interface, allowing users to upload lung CT scan images and receive real-time predictions indicating whether the scan is **Cancerous** or **Non-Cancerous**.

---

## 🚀 Features
- Upload lung CT scan images directly in the Streamlit web app.
- Real-time prediction with confidence score.
- Preprocessing pipeline including image resizing, normalization, and augmentation.
- Trained CNN model with high accuracy for binary classification.
- User-friendly interface for non-technical users.

---

## 🛠 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Pillow (PIL)**

---

## 📂 Project Structure
```

📦 Lung-Cancer-Detection
┣ 📂 dataset              # CT scan dataset (train, test, validation)
┣ 📂 model                # Saved trained CNN model (.h5)
┣ 📂 notebooks            # Jupyter notebooks for model training and testing
┣ 📜 app.py                # Streamlit application script
┣ 📜 train\_model.py        # Model training script
┣ 📜 requirements.txt      # Python dependencies
┗ 📜 README.md             # Project documentation

````

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/<your-username>/lung-cancer-detection.git
cd lung-cancer-detection
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run Streamlit App**

```bash
streamlit run app.py
```

4. **Upload Image & Predict**

* Open the given local URL from the terminal in your browser.
* Upload a lung CT scan image.
* View the prediction result and confidence score.

---

## 📊 Model Training Overview

* Dataset split into **Train**, **Validation**, and **Test** sets.
* Applied **ImageDataGenerator** for preprocessing and augmentation.
* CNN architecture built using **Conv2D**, **MaxPooling2D**, **Flatten**, and **Dense** layers.
* Compiled with `binary_crossentropy` loss and **Adam optimizer**.
* Achieved high accuracy on test data.

---

## 📌 Future Enhancements

* Support for multi-class lung disease classification.
* Integration with cloud-based medical databases.
* Enhanced visualization of detected affected areas.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

**Mahathi Patel**
GitHub: [mahathi](https://github.com/mahathi)
