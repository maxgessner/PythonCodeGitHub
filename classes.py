from Tkinter import Frame


class Dialog2():
    # from Tkinter import Tk, Label, Entry, END, Button, W, mainloop

    def __init__(self, master):
        from Tkinter import Tk, Label, Entry, END, Button, E, mainloop
        import sys

        self.master = master
        master.title('Please enter values for calculation')

        self.label_cross = Label(master, text='crosssection:').grid(row=0)
        Label(master, text='density:').grid(row=1)
        Label(master, text='perimeter:').grid(row=2)
        Label(master, text='length:').grid(row=3)
        Label(master, text='ambient temperature:').grid(row=4)

        self.e1 = Entry(master)
        self.e2 = Entry(master)
        self.e3 = Entry(master)
        self.e4 = Entry(master)
        self.e5 = Entry(master)

        # default values:
        crosssection_initial = 10
        density_initial = 10000
        perimeter_initial = 22
        length_initial = 10
        ambient_temperature_initial = 296.15

        self.e1.insert(END, crosssection_initial)
        self.e2.insert(END, density_initial)
        self.e3.insert(END, perimeter_initial)
        self.e4.insert(END, length_initial)
        self.e5.insert(END, ambient_temperature_initial)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.e4.grid(row=3, column=1)
        self.e5.grid(row=4, column=1)

        def combine_funcs(*funcs):
            def combined_func(*args, **kwargs):
                for f in funcs:
                    f(*args, **kwargs)
            return combined_func

        # self.show_button = Button(master, text='Show',
        # command=self.show_entry_fields) \
        # .grid(row=5, column=0, sticky=W, pady=4)
        self.close_button = Button(master, text='OK',
                                   command=combine_funcs(self.
                                                         show_entry_fields,
                                                         master.destroy))
        self.close_button.grid(row=5, column=0, sticky=E, pady=4)
        self.quit_button = Button(master, text='Quit', command=sys.exit)
        self.quit_button.grid(row=5, column=1, sticky=E, pady=4)
        # self.close_button.bind('<Return>', self.close_button)

    def show_entry_fields(self):

        # default values:
        crosssection_initial = 10
        density_initial = 10000
        perimeter_initial = 22
        length_initial = 10
        ambient_temperature_initial = 296.15

        global crosssection, density, perimeter, length, ambient_temperature
        try:
            crosssection = float(self.e1.get())
        except ValueError:
            crosssection = crosssection_initial
        try:
            density = float(self.e2.get())
        except ValueError:
            density = density_initial
        try:
            perimeter = float(self.e3.get())
        except ValueError:
            perimeter = perimeter_initial
        try:
            length = float(self.e4.get())
        except ValueError:
            length = length_initial
        try:
            ambient_temperature = float(self.e5.get())
        except ValueError:
            ambient_temperature = ambient_temperature_initial
        # print('crosssection: \t\t\t', crosssection, 'mm^2', '\n', \
        #     'density: \t\t\t\t', density, 'g/mm^2', '\n', \
        #     'perimeter: \t\t\t', perimeter, 'mm', '\n', \
        #     'length: \t\t\t', length, 'mm', '\n', \
        #     'ambient temperature: \t', ambient_temperature, 'K')
        # print([crosssection, density, perimeter,
        #        length, ambient_temperature])
        return [crosssection, density, perimeter, length, ambient_temperature]


# check = {}

# irgendwas ist da noch im argen
# der Zugriff auf "calc" vom Inneren der Funktion ist nicht zielfuehrend
# vielleicht laesst sich auch die "Checkbarlist" irgendwie uebergeben?


class Checkbar(Frame):
    from Tkinter import TOP, W
    global check
    check = {}

    def __init__(self, parent=None, picks=[], side=TOP, anchor=W):
        Frame.__init__(self, parent)
        from Tkinter import IntVar, Checkbutton, Button, BOTTOM, TOP
        from Tkinter import RIGHT, YES
        import sys
        self.vars = []
        for pick in picks:
            var = IntVar()
            var.set(1)  # change to 1 to set all checkboxes to true at start
            chk = Checkbutton(self, text=pick, variable=var)
            # chk.pack(side=side, anchor=anchor, expand=YES)
            chk.pack(side=TOP, anchor=anchor, expand=YES)
            self.vars.append(var)
            # print(type(self.vars))
            self.vars[0].set(0)

        def combine_funcs(*funcs):
            def combined_func(*args, **kwargs):
                for f in funcs:
                    f(*args, **kwargs)
            return combined_func

        def allstates(printit):
            # print(list(calc.state()))
            # global check
            # check = list(calc.state())
            # check = {}

            for pick in picks:
                # check[pick] = calc.state()[picks.index(pick)]
                # print(picks.index(pick))
                check[pick] = self.vars[picks.index(pick)].get()
            if printit == 1:
                print(check)
            return check

        Buttonframe = Frame(self)
        Button(Buttonframe, text='QUIT', command=sys.exit).pack(side=RIGHT)
        Button(Buttonframe, text='OK',
               command=combine_funcs(lambda: allstates(0),
                                     parent.destroy)).pack(side=RIGHT)
        Buttonframe.pack(side=BOTTOM)
        # Button(root, text='Peek',
        #          command= lambda: allstates(1)).pack(side=RIGHT)

    def state(self):
        return map((lambda var: var.get()), self.vars)

    def __call__(self, picks=[]):
        for pick in picks:
            # check[pick] = calc.state()[picks.index(pick)]
            # print(picks.index(pick))
            check[pick] = self.vars[picks.index(pick)].get()
        return check
