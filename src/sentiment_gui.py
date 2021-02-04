import tkinter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from keyword_extraction import KeywordExtraction
from utils.visualization import Visualization


class SentimentGUI:
    def __init__(self, df_paragraphs, df_term_wights):
        self.df_paragraphs = df_paragraphs
        self.df_term_weights = df_term_wights
        self.plots = []
        self.current_plot = None
        self.current_plot_index = 0
        self.gui = None
        self.next_button = None
        self.previous_button = None
        pass

    def clear_plots(self, clear_plot_array=False):
        if self.current_plot is not None:
            self.current_plot.get_tk_widget()["command"] = self.current_plot.get_tk_widget().forget()
        if clear_plot_array:
            self.plots = []

    def show_diagram(self, first_image=False, increase=True):
        if not first_image and self.current_plot_index + 1 < len(self.plots) and increase:
            self.current_plot_index += 1
        elif not first_image and self.current_plot_index + 1 >= len(self.plots) and increase:
            return
        elif not first_image and self.current_plot_index > 0 and not increase:
            self.current_plot_index -= 1
        if self.current_plot_index + 1 == len(self.plots):
            self.next_button["state"] = "disable"
        else:
            self.next_button["state"] = "normal"
        if self.current_plot_index == 0:
            self.previous_button["state"] = "disable"
        else:
            self.previous_button["state"] = "normal"
        self.clear_plots()
        self.current_plot = self.plots[self.current_plot_index]
        self.current_plot.get_tk_widget().pack(fill=tkinter.BOTH)

    def show_sentiment(self, by_party):
        self.clear_plots(clear_plot_array=True)
        self.current_plot_index = 0
        figures = Visualization.show_pie_charts(self.df_paragraphs, by_party=by_party)
        for fig in figures:
            bar1 = FigureCanvasTkAgg(fig, self.gui)
            self.plots.append(bar1)
        self.show_diagram(first_image=True)
        self.next_button["state"] = "normal"

    def show_topics(self):
        self.next_button["state"] = "disabled"
        self.previous_button["state"] = "disabled"
        self.clear_plots(clear_plot_array=True)
        keyword_extraction = KeywordExtraction(self.df_paragraphs)
        fig = keyword_extraction.show_graph(self.df_term_weights)
        self.current_plot = FigureCanvasTkAgg(fig, self.gui)
        self.current_plot.get_tk_widget().pack(fill=tkinter.BOTH)

    def iterate_plot(self):
        self.show_diagram()

    def show_gui(self):
        self.gui = tkinter.Tk()
        self.gui.geometry("1000x1000")
        # self.gui.geometry("%dx%d" % (self.gui.winfo_screenwidth(), self.gui.winfo_screenheight()))

        button_by_party = tkinter.Button(
            self.gui, text="Sentiment by party", command=lambda: self.show_sentiment(by_party=True)
        )
        button_by_party.pack(side=tkinter.TOP)

        button_by_outlet = tkinter.Button(
            self.gui, text="Sentiment by outlet", command=lambda: self.show_sentiment(by_party=False)
        )
        button_by_outlet.pack(side=tkinter.TOP)

        button_topic = tkinter.Button(self.gui, text="Show topics", command=self.show_topics)
        button_topic.pack(side=tkinter.TOP)

        self.next_button = button_by_outlet = tkinter.Button(
            self.gui, text="Show next", command=lambda: self.show_diagram(increase=True)
        )
        self.next_button["state"] = "disabled"
        button_by_outlet.pack(side=tkinter.BOTTOM)

        self.previous_button = button_by_outlet = tkinter.Button(
            self.gui, text="Show previous", command=lambda: self.show_diagram(increase=False)
        )
        self.previous_button["state"] = "disabled"
        button_by_outlet.pack(side=tkinter.BOTTOM)

        # "Hack" for displaying topics correctly, otherwise they sometimes appear in pie charts
        self.show_topics()
        self.clear_plots()

        self.gui.mainloop()
