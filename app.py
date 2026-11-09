"""Local desktop demo using Python's standard tkinter GUI."""
import tkinter as tk
from tkinter import ttk
from predict import load_model, classify


def main():
    model = load_model()
    root = tk.Tk()
    root.title('NLP Mini Project | SMS Spam Detector')
    root.geometry('640x420')
    root.minsize(500, 360)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='SMS Spam Detector', font=('Segoe UI', 22, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='TF-IDF + Multinomial Naive Bayes').pack(anchor='w', pady=(4, 18))
    entry = tk.Text(frame, height=7, wrap='word', font=('Segoe UI', 12))
    entry.pack(fill='both', expand=True)
    result = tk.StringVar(value='Enter a message, then select Classify.')

    def submit():
        try:
            label, score = classify(model, entry.get('1.0', 'end'))
            result.set(f'{"SPAM" if label == "spam" else "HAM (not spam)"}\nModel spam probability: {score:.1%}')
        except ValueError as error:
            result.set(str(error))

    ttk.Button(frame, text='Classify message', command=submit).pack(anchor='w', pady=14)
    ttk.Label(frame, textvariable=result, font=('Segoe UI', 12)).pack(anchor='w')
    ttk.Label(frame, text='Educational model: predictions can be wrong; scores are not calibrated.').pack(anchor='w', pady=(14, 0))
    root.mainloop()


if __name__ == '__main__':
    main()
