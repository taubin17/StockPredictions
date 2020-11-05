import tkinter as tk
from pathlib import Path
import os
# import sys
import Grabber
import ModelMaker
import PredictNewPrices
import tensorflow as tf
import sys
import LineAdjustTool

# Get user name to navigate program save location
Username = os.getlogin()

symbols = []
Symbols_Directory = Path('C:/Users/' + Username + '/Documents/StockTrader/')


def create_home_directory():
    if not os.path.exists(Symbols_Directory):
        os.makedirs(Symbols_Directory)


def main():
    create_home_directory()

    root = tk.Tk()

    # Labels for walking a user through the program
    label1 = tk.Label(text='Greetings! I Am Tyler, the creator and developer of this software!')
    label2 = tk.Label(text='In the text box below, please enter stock symbols, hit done when symbol is typed')
    label3 = tk.Label(text='And hit the ready button to proceed to market analysis')
    label4 = tk.Label(
        text='To find all documents, charts, etc. on your stocks, go to documents and find the StockTrader folder')

    # Assign label locations
    label1.grid(row=0, column=0)
    label2.grid(row=1, column=0)
    label3.grid(row=2, column=0)
    label4.grid(row=3, column=0)

    # Create a button to exit GUI when done
    button1 = tk.Button(root, text='Finished', command=lambda: run_symbols(root))
    button1.grid(row=4, column=1)

    # Create a container to contain any entered user text
    text = tk.StringVar()
    e1 = tk.Entry(root, textvariable=text)

    # Assign the container location
    e1.grid(row=4, column=0)

    # Once enter is hit, add to the list of symbols entered (Lambda takes care of that)
    root.bind('<Return>', lambda event: entry_clear(e1, text))

    root.mainloop()


def run_symbols(root):
    file_path = Symbols_Directory / 'symbols.txt'
    fd = open(file_path, 'w+')
    for each in symbols:
        fd.write((each.upper()) + '\n')

    fd.close()

    # Flush must be called in order for each script to possess necessary info at the time of start, otherwise
    # dependency error
    sys.stdout.flush()
    root.destroy()

    # Grab, model, and predict the stock data from Apache API
    paths = Grabber.main()
    LineAdjustTool.main(paths)

    ModelMaker.main(paths)
    PredictNewPrices.main(paths)

    # Destroy the program
    sys.exit()


# Legacy function, does not contribute to program as of yet
def pressed_key(event):
    return event.char


# Remove text in text box to get new symbol data
def entry_clear(e1, text):

    global symbols

    # Get whatever is in textbox, and try to find it
    new_symbol = text.get()

    # Get rid of text in box so it appears empty again
    e1.delete(0, 'end')

    # Add new symbol to the list
    symbols.append(new_symbol)
    return symbols


if __name__ == '__main__':
    main()
