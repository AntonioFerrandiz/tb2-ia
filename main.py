import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, ttk, PhotoImage
from PIL import ImageTk, Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

root = tk.Tk()
root.title("Proyecto 2 - Inteligencia Artificial")
root.geometry("800x800")
root.pack_propagate(False)
root.resizable(1, 1)
root.grid_rowconfigure(0, weight=0)
root.grid_columnconfigure(0, weight=1)

bg = PhotoImage(file="assets/img/bg.png")
background_label= tk.Label(root,image=bg)
background_label.place(x=0,y=0,relwidth=1,relheight=1)

frame1 = tk.LabelFrame(root, text="Dataset")
frame1.place(height=500, width=800)


diabetes_dataset = pd.read_csv('assets/datasets/diabetes.csv')
pd.read_csv

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    #print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)



file_frame = tk.LabelFrame(root, text="Opciones")
file_frame.configure(background='#f9fcfb')
file_frame.place(height=100, width=600, rely=0.65, relx=0, x=100)

button1 = tk.Button(file_frame, text="Cargar dataset", command=lambda: Load_dataset())
button1.place(rely=0.25, relx=0.05)

button2 = tk.Button(file_frame, text="Información general", command=lambda: get_information())
button2.place(rely=0.25, relx=0.25)

button3 = tk.Button(file_frame, text="Entrenar data", command=lambda: train_data())
button3.place(rely=0.25, relx=0.50)

button4 = tk.Button(file_frame, text="Hacer predicción", command=lambda: make_prediction())
button4.place(rely=0.25, relx=0.75)


tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

def Load_dataset():
    file_path = 'assets/datasets/diabetes.csv'

    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        tk.messagebox.showerror("Information", "El archivo seleccionado, no es valido")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f" {file_path}")

        return None

    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row)
def dataset_outcome():
    w_outcome = Toplevel(w_information)
    w_outcome.geometry("200x150")
    w_outcome.title("Dataset outcome")
    w_outcome.resizable(False, False)
    
    dataset_outcome_info = diabetes_dataset['Outcome'].value_counts()
    lbl001 = tk.Label(w_outcome, text=dataset_outcome_info)
    lbl001.pack()

    lbl002 = tk.Label(w_outcome, text="1: Pacientes con diabetes")
    lbl002.pack()
    lbl003 = tk.Label(w_outcome, text="0: Pacientes sin diabetes")
    lbl003.pack()

    close_btn = Button(w_outcome, text="Cerrar", command=lambda: w_outcome.destroy())
    close_btn.pack()
def dataset_mean():
    w_mean = Toplevel(w_information)
    w_mean.geometry("750x100")
    w_mean.title("Dataset mean")
    w_mean.resizable(False, False)
    
    dataset_mean_info = diabetes_dataset.groupby('Outcome').mean()
    lbl001 = tk.Label(w_mean, text=dataset_mean_info)
    lbl001.pack()

    close_btn = Button(w_mean, text="Cerrar", command=lambda: w_mean.destroy())
    close_btn.pack()
def get_information():
    global w_information
    w_information = Toplevel(root)
    w_information.geometry("250x250")
    w_information.title("Dataset information")
    w_information.resizable(False, False)

    dataset_size = diabetes_dataset.shape
    lbl001 = tk.Label(w_information, text=f"Tamaño de dataset: {dataset_size}")
    lbl001.pack()

    dataset_outcome_btn = tk.Button(w_information, text="Dataset Outcome", command=lambda: dataset_outcome())
    dataset_outcome_btn.pack()

    dataset_mean_btn = tk.Button(w_information, text="Dataset Mean", command=lambda: dataset_mean())
    dataset_mean_btn.pack()

    close_btn = Button(w_information, text="Cerrar", command=lambda: w_information.destroy())
    close_btn.pack()
def train_data():
    w_train = Toplevel(root)
    w_train.geometry("450x100")
    w_train.title("Dataset train")
    w_train.resizable(False, False)
    

    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    lbl001 = tk.Label(w_train, text=f"Puntuación de precisión de los datos de entrenamiento: {training_data_accuracy}")
    lbl001.pack()

    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    lbl002 = tk.Label(w_train, text=f"Puntuación de precisión de los datos de la prueba: {test_data_accuracy}")
    lbl002.pack()
    
    close_btn = Button(w_train, text="Cerrar", command=lambda: w_train.destroy())
    close_btn.pack()
def make_prediction():
    w_prediction = Toplevel(root)
    w_prediction.geometry("450x500")
    w_prediction.title("Dataset train")
    w_prediction.resizable(False, False)

    lbl001 = tk.Label(w_prediction, text="Pregnancies")
    lbl001.pack()
    entry001 = tk.Entry(w_prediction, width=10)
    entry001.pack()

    lbl002 = tk.Label(w_prediction, text="Glucose")
    lbl002.pack()
    entry002 = tk.Entry(w_prediction, width=10)
    entry002.pack()

    lbl003 = tk.Label(w_prediction, text="BloodPressure")
    lbl003.pack()
    entry003 = tk.Entry(w_prediction, width=10)
    entry003.pack()

    lbl004 = tk.Label(w_prediction, text="SkinThickness")
    lbl004.pack()
    entry004 = tk.Entry(w_prediction, width=10)
    entry004.pack()

    lbl005 = tk.Label(w_prediction, text="Insulin")
    lbl005.pack()
    entry005 = tk.Entry(w_prediction, width=10)
    entry005.pack()

    lbl006 = tk.Label(w_prediction, text="BMI")
    lbl006.pack()
    entry006 = tk.Entry(w_prediction, width=10)
    entry006.pack()

    lbl007 = tk.Label(w_prediction, text="PedigreeFunction")
    lbl007.pack()
    entry007 = tk.Entry(w_prediction, width=10)
    entry007.pack()

    lbl008 = tk.Label(w_prediction, text="Age")
    lbl008.pack()
    entry008 = tk.Entry(w_prediction, width=10)
    entry008.pack()

    lbl009 = tk.Label(w_prediction, text="Tiene diabetes?")
    lbl009.pack()

    lbl010 = tk.Label(w_prediction, text="....")
    lbl010.pack()
    def predict_diabetes():
        input_data = (float(entry001.get()), float(entry002.get()), float(entry003.get()),
        float(entry004.get()), float(entry005.get()), float(entry006.get()),
        float(entry007.get()), float(entry008.get()))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = classifier.predict(std_data)
        if (prediction[0] == 0):
            lbl010["text"] = "No"
        else:
            lbl010["text"] = "Si"
    calculate_btn = Button(w_prediction, text="Prediction", command=lambda: predict_diabetes())
    calculate_btn.pack()
    # input_data = (5,166,72,19,175,25.8,0.587,51)
   

root.mainloop()