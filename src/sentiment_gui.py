import tkinter
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from keyword_extraction import KeywordExtraction
from utils.visualization import Visualization


class SentimentGUI:
    def __init__(self, df_paragraphs):
        self.df_paragraphs = df_paragraphs
        self.df_paragraphs_configured = df_paragraphs.copy()
        self.keyword_extraction = KeywordExtraction(self.df_paragraphs_configured)
        self.plots = []
        self.current_plot = None
        self.current_plot_index = 0
        self.gui = None
        self.next_button = None
        self.previous_button = None
        self.date_check = None
        self.cdu_check = None
        self.csu_check = None
        self.spd_check = None
        self.afd_check = None
        self.gruene_check = None
        self.linke_check = None
        self.tagesschau_check = None
        self.taz_check = None
        self.bild_check = None
        self.entry_date_from = None
        self.entry_date_to = None
        pass

    def configure_dataframe(self):
        self.df_paragraphs_configured = self.df_paragraphs
        if self.date_check.get() == 1:
            self.filter_time(self.entry_date_from.get(), self.entry_date_to.get())

    def clear_plots(self, clear_plot_array=False):
        if self.current_plot is not None:
            self.current_plot.get_tk_widget()["command"] = self.current_plot.get_tk_widget().grid_forget()
        if clear_plot_array:
            self.plots = []

    def show_diagram(self, first_image=False, increase=True):
        plt.close("all")
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
        self.current_plot.get_tk_widget().grid(row=4, column=0, columnspan=3)

    def get_parties(self):
        party_list = []
        if self.cdu_check.get() == 1:
            party_list.append("CDU")
        if self.csu_check.get() == 1:
            party_list.append("CSU")
        if self.spd_check.get() == 1:
            party_list.append("SPD")
        if self.afd_check.get() == 1:
            party_list.append("AfD")
        if self.gruene_check.get() == 1:
            party_list.append("Grüne")
        if self.linke_check.get() == 1:
            party_list.append("Linke")
        return party_list

    def get_media(self):
        media_list = []
        if self.tagesschau_check.get() == 1:
            media_list.append("Tagesschau")
        if self.taz_check.get() == 1:
            media_list.append("TAZ")
        if self.bild_check.get() == 1:
            media_list.append("Bild")
        return media_list

    def show_sentiment(self, by_party):
        plt.close("all")
        self.clear_plots(clear_plot_array=True)
        self.current_plot_index = 0
        party_list = self.get_parties()
        media_list = self.get_media()
        self.configure_dataframe()
        figures = Visualization.show_pie_charts(
            self.df_paragraphs_configured, by_party=by_party, parties=party_list, media=media_list
        )
        for fig in figures:
            bar1 = FigureCanvasTkAgg(fig, self.gui)
            self.plots.append(bar1)
        self.show_diagram(first_image=True)
        self.next_button["state"] = "normal"

    def show_topics(self):
        self.next_button["state"] = "disabled"
        self.previous_button["state"] = "disabled"
        self.clear_plots(clear_plot_array=True)
        self.configure_dataframe()
        self.keyword_extraction.set_data(self.df_paragraphs_configured)
        party_list = self.get_parties()
        df_term_weights = self.keyword_extraction.get_term_weight_tuples(parties=party_list)
        fig = self.keyword_extraction.show_graph(df_term_weights)
        self.current_plot = FigureCanvasTkAgg(fig, self.gui)
        self.current_plot.get_tk_widget().grid(row=4, column=0, columnspan=3)

    def iterate_plot(self):
        self.show_diagram()

    def filter_time(self, min_date=None, max_date=None):
        self.df_paragraphs_configured = self.df_paragraphs_configured[self.df_paragraphs_configured["date"].notna()]
        if min_date and max_date:
            self.df_paragraphs_configured = self.df_paragraphs_configured[
                (self.df_paragraphs_configured["date"] > min_date) & (self.df_paragraphs_configured["date"] < max_date)
            ]

    def enable_date_setting(self):
        if self.date_check.get() == 1:
            self.filter_time()
            self.entry_date_from.insert(tkinter.END, min(self.df_paragraphs_configured["date"]))
            self.entry_date_to.insert(tkinter.END, max(self.df_paragraphs_configured["date"]))
        else:
            self.df_paragraphs_configured = self.df_paragraphs
            self.entry_date_from.delete(0, "end")
            self.entry_date_to.delete(0, "end")

    def show_gui(self):
        self.gui = tkinter.Tk()
        self.gui.geometry("1000x1000")
        # self.gui.geometry("%dx%d" % (self.gui.winfo_screenwidth(), self.gui.winfo_screenheight()))
        self.date_check = tkinter.IntVar(value=0)
        self.cdu_check = tkinter.IntVar(value=1)
        self.csu_check = tkinter.IntVar(value=1)
        self.spd_check = tkinter.IntVar(value=1)
        self.afd_check = tkinter.IntVar(value=1)
        self.gruene_check = tkinter.IntVar(value=1)
        self.linke_check = tkinter.IntVar(value=1)
        self.tagesschau_check = tkinter.IntVar(value=1)
        self.taz_check = tkinter.IntVar(value=1)
        self.bild_check = tkinter.IntVar(value=1)

        button_by_party = tkinter.Button(
            self.gui, text="Sentiment by party", command=lambda: self.show_sentiment(by_party=True)
        )
        button_by_party.grid(row=0, column=0)

        button_by_outlet = tkinter.Button(
            self.gui, text="Sentiment by outlet", command=lambda: self.show_sentiment(by_party=False)
        )
        button_by_outlet.grid(row=0, column=1)

        button_topic = tkinter.Button(self.gui, text="Show topics", command=self.show_topics)
        button_topic.grid(row=0, column=2)

        check_filter_date = tkinter.Checkbutton(
            self.gui,
            text="Filter dates",
            variable=self.date_check,
            onvalue=1,
            offvalue=0,
            command=self.enable_date_setting,
        )
        check_filter_date.grid(row=1, column=0)
        label_date_from = tkinter.Label(self.gui, text="Date: From ")
        label_date_from.grid(row=1, column=1)
        self.entry_date_from = tkinter.Entry(self.gui, bd=5)
        # entry_date_from.insert(tkinter.END, min(self.df_paragraphs["date"].to_list()))
        self.entry_date_from.grid(row=1, column=2)

        label_date_to = tkinter.Label(self.gui, text=" To ")
        label_date_to.grid(row=1, column=3)
        self.entry_date_to = tkinter.Entry(self.gui, bd=5)
        # entry_date_to.insert(tkinter.END, max(self.df_paragraphs["date"]))
        self.entry_date_to.grid(row=1, column=4)

        check_cdu = tkinter.Checkbutton(self.gui, text="CDU", variable=self.cdu_check, onvalue=1, offvalue=0)
        check_cdu.grid(row=2, column=0)
        check_csu = tkinter.Checkbutton(self.gui, text="CSU", variable=self.csu_check, onvalue=1, offvalue=0)
        check_csu.grid(row=2, column=1)
        check_spd = tkinter.Checkbutton(self.gui, text="SPD", variable=self.spd_check, onvalue=1, offvalue=0)
        check_spd.grid(row=2, column=2)
        check_afd = tkinter.Checkbutton(self.gui, text="AfD", variable=self.afd_check, onvalue=1, offvalue=0)
        check_afd.grid(row=2, column=3)
        check_gruene = tkinter.Checkbutton(self.gui, text="Grüne", variable=self.gruene_check, onvalue=1, offvalue=0)
        check_gruene.grid(row=2, column=4)
        check_linke = tkinter.Checkbutton(self.gui, text="Linke", variable=self.linke_check, onvalue=1, offvalue=0)
        check_linke.grid(row=2, column=5)

        check_tagesschau = tkinter.Checkbutton(
            self.gui, text="Tagesschau", variable=self.tagesschau_check, onvalue=1, offvalue=0
        )
        check_tagesschau.grid(row=3, column=0)
        check_taz = tkinter.Checkbutton(self.gui, text="TAZ", variable=self.taz_check, onvalue=1, offvalue=0)
        check_taz.grid(row=3, column=1)
        check_bild = tkinter.Checkbutton(self.gui, text="Bild", variable=self.bild_check, onvalue=1, offvalue=0)
        check_bild.grid(row=3, column=2)

        self.next_button = button_by_outlet = tkinter.Button(
            self.gui, text="Show next", command=lambda: self.show_diagram(increase=True)
        )
        self.next_button["state"] = "disabled"
        button_by_outlet.grid(row=5, column=2)

        self.previous_button = button_by_outlet = tkinter.Button(
            self.gui, text="Show previous", command=lambda: self.show_diagram(increase=False)
        )
        self.previous_button["state"] = "disabled"
        button_by_outlet.grid(row=5, column=0)

        # "Hack" for displaying topics correctly, otherwise they sometimes appear in pie charts
        self.show_topics()
        self.clear_plots()

        self.gui.mainloop()
