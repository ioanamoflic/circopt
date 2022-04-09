from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
import cirq
from ambiguous_evaluation import optimize
import circopt_utils as utils

ws = Tk()
ws.title('Optimising circuits')
ws.geometry('650x500')


class CircuitUploader:

    def __init__(self):
        self.circuit = cirq.Circuit()
        self.initial_gate_count: int = 0
        self.final_gate_count: int = 0
        self.initial_length: int = 0
        self.final_length: int = 0
        self.moment_range: int = 5

    def _get_gate_count(self, circuit) -> int:
        counter: int = 0
        for moment in circuit:
            counter += len(moment)
        return counter

    def open_file(self):
        file_path = askopenfile(mode='r', filetypes=[('.txt file', '*.txt')])
        if file_path is not None:
            f = open(file_path.name, 'r')
            json_string = f.read()
            self.circuit = cirq.read_json(json_text=json_string)
            self.initial_gate_count = self._get_gate_count(self.circuit)
            self.initial_length = len(cirq.Circuit(self.circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST))

            txtStartCircuit = Text(ws, width=75, height=10)
            txtStartCircuit.grid(column=0, row=5)
            txtStartCircuit.insert(END, str(self.circuit))

            initialGateCount = Label(
                ws,
                text=f'Initial gate count: {circuitUploader.initial_gate_count}'
            )
            initialGateCount.grid(row=1, column=0, padx=10)

            initialLength = Label(
                ws,
                text=f'Initial length: {circuitUploader.initial_length}'
            )
            initialLength.grid(row=2, column=0, padx=10)

    def optimize(self):
        pb1 = Progressbar(
            ws,
            orient=HORIZONTAL,
            length=300,
            mode='determinate'
        )
        pb1.grid(row=4, columnspan=3, pady=20)

        q, s, a = utils.read_train_data()

        final_circuit = optimize(self.circuit, q, s, a, self.moment_range, steps=1)
        final_gate_count = self._get_gate_count(final_circuit)
        final_len = len(cirq.Circuit(final_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST))

        txtFinalCircuit = Text(ws, width=75, height=10)
        txtFinalCircuit.grid(column=0, row=6)
        txtFinalCircuit.insert(END, str(final_circuit))

        finalGateCount = Label(
            ws,
            text=f'Final gate count: {final_gate_count}'
        )
        finalGateCount.grid(row=3, column=0, padx=10)

        finalLength = Label(
            ws,
            text=f'Final length: {final_len}'
        )
        finalLength.grid(row=4, column=0, padx=10)

        for i in range(5):
            ws.update_idletasks()
            pb1['value'] += 20
            time.sleep(1)
        pb1.destroy()
        Label(ws, text='Optimization finished!', foreground='green').grid(row=7, columnspan=3, pady=10)


circuitUploader = CircuitUploader()

uploadCircuitLabel = Label(
    ws,
    text='Upload circuit txt file.'
)
uploadCircuitLabel.grid(row=0, column=0, padx=10)

uploadCircuitLabel = Label(
    ws,
    text='Upload circuit txt file.'
)
uploadCircuitLabel.grid(row=0, column=0, padx=10)

uploadCircuitBtn = Button(
    ws,
    text='Choose File',
    command=lambda: circuitUploader.open_file()
)
uploadCircuitBtn.grid(row=0, column=1)


optimizeBtn = Button(
    ws,
    text='Optimize',
    command=lambda: circuitUploader.optimize()
)
optimizeBtn.grid(row=3, column=1)

ws.mainloop()