import datetime
import tkinter
import webbrowser
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame

from keyword_extraction import KeywordExtraction
from model.plot_type import PlotType
from utils.visualization import Visualization


class SentimentGUI:
    """
    Class that creates a GUI to visualize the results of the news analysis with different filter criteria
    """

    def __init__(self, df_paragraphs: DataFrame):
        """
        :param df_paragraphs: processed dataframe with analysis results
        """
        self.df_paragraphs = df_paragraphs
        self.df_paragraphs_configured = df_paragraphs.copy()
        self.keyword_extraction = KeywordExtraction(self.df_paragraphs_configured)
        self.plots = []
        self.current_plot = None
        self.current_plot_index = 0
        self.help_messages = {
            PlotType.SENTIMENT_PARTY: "These are piecharts showing the sentiment towards a certain party\n of the "
            "selected media. This can be either positive, negative or neutral\n "
            '(see legend). With a click on "Show next" or "Show previous"\n you can '
            "see the sentiment for other parties.",
            PlotType.SENTIMENT_OUTLET: "These are piecharts showing the sentiment of a certain media\n towards the "
            "selected parties. This can be either positive, negative or neutral\n "
            '(see legend). With a click on "Show next" or "Show previous"\n you can '
            "see the sentiment for other media.",
            PlotType.TOPICS_PARTY: "This is a bipartite graph showing the most important topics for the selected "
            "parties.\n On the left side you can see the parties and on the right side the "
            "the corresponding terms.\n The thickness of the connecting lines are the term "
            "counts\n how often the term appears for each party.",
            PlotType.TOPICS_MEDIA: "This is a bipartite graph showing the most important topics for the selected "
            "media.\n On the left side you can see the media and on the right side the "
            "the corresponding terms.\n The thickness of the connecting lines are the term "
            "counts\n how often the term appears for each party.",
            PlotType.TIME_COURSE: "This is a line graph showing the importance of a certain term in the selected "
            'media.\n With a click on "Show next" or "Show previous"\n'
            "you can see the importance of another term.",
        }
        self.current_plot_type = None
        self.gui = None
        self.next_button = None
        self.previous_button = None
        self.help_button = None
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

    def configure_dataframe(self) -> None:
        """
        if a time filter is set in GUI, only articles within this timespan are considered.
        Other articles are filtered out in this method
        """
        self.df_paragraphs_configured = self.df_paragraphs
        if self.date_check.get() == 1:
            self.filter_time(self.entry_date_from.get(), self.entry_date_to.get())

    def filter_time(self, min_date: str = None, max_date: str = None) -> None:
        """
        Filer data frame to contain only articles from a certain time period
        :param min_date: start date of the time period
        :param max_date: end date of the time period
        """
        self.df_paragraphs_configured = self.df_paragraphs_configured[self.df_paragraphs_configured["date"].notna()]
        if min_date and max_date:
            self.df_paragraphs_configured = self.df_paragraphs_configured[
                (self.df_paragraphs_configured["date"] > min_date) & (self.df_paragraphs_configured["date"] < max_date)
            ]

    def configure_dataframe_for_time_course(self, start_date, end_date, media):
        df_paragraphs_time_interval = self.df_paragraphs_configured[self.df_paragraphs_configured["date"].notna()]
        df_paragraphs_time_interval = df_paragraphs_time_interval[df_paragraphs_time_interval["media"] == media]
        if start_date and end_date:
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                (df_paragraphs_time_interval["date"] > start) & (df_paragraphs_time_interval["date"] < end)
            ]
        return df_paragraphs_time_interval

    def clear_plots(self, clear_plot_array: bool = False) -> None:
        """
        Clears the plot in the GUI and optionally also the array with all current plots

        :param clear_plot_array: True if all current plots in the plots should be deleted (set when changing category
                e.g. from sentiment to topics that only plots from current category are in plot array)
        """
        self.help_button["state"] = "disabled"
        if self.current_plot is not None:
            self.current_plot.get_tk_widget()["command"] = self.current_plot.get_tk_widget().grid_forget()
        if clear_plot_array:
            self.plots = []

    def show_diagram(self, first_image: bool = False, increase: bool = True) -> None:
        """
        Shows the plot in the plots array at current plot index and optionally increases/ decreases plot index before
        :param first_image: True if the first diagram should be shown (e.g. when initially displaying sentiment)
        :param increase: True if next plot should be shown (button "show next"), False for previous plot
        """
        plt.close("all")
        # increase if there are still next plots left
        if not first_image and self.current_plot_index + 1 < len(self.plots) and increase:
            self.current_plot_index += 1
        # if there is no next image in plots, return
        elif not first_image and self.current_plot_index + 1 >= len(self.plots) and increase:
            return
        # decrease if there are previous plots to show
        elif not first_image and self.current_plot_index > 0 and not increase:
            self.current_plot_index -= 1
        # if after increasing the index is the last plot in plots array, disable next button
        if self.current_plot_index + 1 == len(self.plots):
            self.next_button["state"] = "disable"
        else:
            self.next_button["state"] = "normal"
        # if after decreasing the index in plots is 0 (first plot), disable previous button
        if self.current_plot_index == 0:
            self.previous_button["state"] = "disable"
        else:
            self.previous_button["state"] = "normal"
        # clear plot in GUI
        self.clear_plots()
        # enable help button
        self.help_button["state"] = "normal"
        # set current plot
        self.current_plot = self.plots[self.current_plot_index]
        self.current_plot.get_tk_widget().grid(row=5, column=0, columnspan=6)

    def get_parties(self) -> List[str]:
        """
        Determines which parties are currently checked (enabled) in GUI
        :return: list with the currently selected parties
        """
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

    def get_media(self) -> List[str]:
        """
        Determines which media are currently checked (enabled) in GUI
        :return: list with the currently selected media
        """
        media_list = []
        if self.tagesschau_check.get() == 1:
            media_list.append("Tagesschau")
        if self.taz_check.get() == 1:
            media_list.append("TAZ")
        if self.bild_check.get() == 1:
            media_list.append("Bild")
        return media_list

    def show_sentiment(self, by_party: bool) -> None:
        """
        shows the pie chart with the sentiment either by party or by media
        :param by_party: if True, shows the sentiment sorted by party, that is a party and then 3 plots with the
                         sentiment of the media toward this party. Otherwise sorted by media, that is a media with the
                         6 plots of the sentiment towards different parties
        """
        plt.close("all")
        self.clear_plots(clear_plot_array=True)
        if by_party:
            self.current_plot_type = PlotType.SENTIMENT_PARTY
        else:
            self.current_plot_type = PlotType.SENTIMENT_OUTLET
        self.current_plot_index = 0
        # get currently enabled parties and media
        party_list = self.get_parties()
        media_list = self.get_media()
        # filter time
        self.configure_dataframe()
        # get the pie charts from visualization class
        figures = Visualization.get_pie_charts(
            self.df_paragraphs_configured, by_party=by_party, parties=party_list, media=media_list
        )
        # get canvas to show in gui from each of the figures and store it in plots array
        for fig in figures:
            bar1 = FigureCanvasTkAgg(fig, self.gui)
            self.plots.append(bar1)
        # display the first diagram
        self.show_diagram(first_image=True)
        self.next_button["state"] = "normal"

    def show_topics(self, by_party) -> None:
        """
        shows a bipartite graph with the important topics/keywords for each party
        """
        if by_party:
            self.current_plot_type = PlotType.TOPICS_PARTY
        else:
            self.current_plot_type = PlotType.TOPICS_MEDIA
        # disable previous and next button as it is only one figure to show
        self.next_button["state"] = "disabled"
        self.previous_button["state"] = "disabled"
        self.clear_plots(clear_plot_array=True)
        # set help text
        self.help_button["state"] = "normal"
        # filter time
        self.configure_dataframe()
        # set the filtered dataframe for the keyword extraction to only include filtered articles
        self.keyword_extraction.set_data(self.df_paragraphs_configured)
        # get currently enabled parties and media
        party_list = self.get_parties()
        media_list = self.get_media()
        # get keywords/ graph for media and parties
        self.keyword_extraction.set_active_media(media_list)
        df_term_weights = self.keyword_extraction.get_term_weight_tuples(
            by_party=by_party, parties=party_list, media=media_list
        )
        fig = self.keyword_extraction.get_graph(df_term_weights)
        # show the plot in GUI
        self.current_plot = FigureCanvasTkAgg(fig, self.gui)
        self.current_plot.get_tk_widget().grid(row=5, column=0, columnspan=6)

    def show_time_course(self):
        self.current_plot_type = PlotType.TIME_COURSE
        self.next_button["state"] = "normal"
        self.previous_button["state"] = "normal"
        self.clear_plots(clear_plot_array=True)
        self.help_button["state"] = "normal"
        self.configure_dataframe()
        self.keyword_extraction.set_data(self.df_paragraphs_configured)
        party_list = self.get_parties()
        media_list = self.get_media()
        self.keyword_extraction.set_active_media(media_list)
        df_top_terms = self.keyword_extraction.get_top_terms_for_party(parties=party_list)
        df_image = pd.DataFrame(columns=["party", "media", "term", "weight"])
        for party in party_list:
            for media in media_list:
                # use only top 3 words for each party
                df_party_term = df_top_terms[df_top_terms["party"] == party]
                for term in df_party_term["term"]:
                    # split time window into smaller chunks
                    # calculate months between dates
                    initial_start_date = datetime.datetime.strptime(self.entry_date_from.get(), "%Y-%m-%d")
                    start_date = initial_start_date
                    initial_end_date = datetime.datetime.strptime(self.entry_date_to.get(), "%Y-%m-%d")
                    next_end_date = initial_start_date + relativedelta(months=+1)
                    weight_list = []
                    dates = []
                    while next_end_date < initial_end_date:
                        # get weight for each month
                        df_interval_paragraphs = self.configure_dataframe_for_time_course(
                            start_date, next_end_date, media
                        )
                        weight = self.keyword_extraction.get_term_count(df_interval_paragraphs, True, party, term)
                        dates.append(start_date)
                        weight_list.append(weight)
                        start_date = next_end_date
                        next_end_date = start_date + relativedelta(months=+1)
                    df_image = df_image.append(
                        {"party": party, "media": media, "term": term, "weight": weight_list, "dates": dates},
                        ignore_index=True,
                    )
        # draw plot for time window
        figures = Visualization.get_plots(df_image)
        for fig in figures:
            bar1 = FigureCanvasTkAgg(fig, self.gui)
            self.plots.append(bar1)
        self.show_diagram(first_image=True)

    def iterate_plot(self):
        self.show_diagram()

    def enable_date_setting(self) -> None:
        """
        Sets the dates in the text fields to the minimum and maximum of the current dataframe if text filtering is
        checked, otherwise clears textfield
        """
        # set date textfields to minimum and maximum of available news data
        if self.date_check.get() == 1:
            self.filter_time()
            self.entry_date_from.insert(tkinter.END, min(self.df_paragraphs_configured["date"]))
            self.entry_date_to.insert(tkinter.END, max(self.df_paragraphs_configured["date"]))
        # clear date textfields
        else:
            self.df_paragraphs_configured = self.df_paragraphs
            self.entry_date_from.delete(0, "end")
            self.entry_date_to.delete(0, "end")

    def popupmsg(self):
        popup = tkinter.Tk()
        popup.geometry("500x200")
        popup.wm_title("Description")
        label = tkinter.Label(popup, text=self.help_messages[self.current_plot_type])
        label.pack(side="top", fill="x", pady=10)
        button = tkinter.Button(popup, text="Okay", command=popup.destroy)
        button.pack()
        popup.mainloop()

    def open_browser(self, url):
        webbrowser.open_new(url)

    def update_gui(self) -> None:
        """
        updates the gui in case a filter criteria has been changed
        """
        if self.current_plot_type == PlotType.SENTIMENT_PARTY:
            self.show_sentiment(by_party=True)
        elif self.current_plot_type == PlotType.SENTIMENT_OUTLET:
            self.show_sentiment(by_party=False)
        elif self.current_plot_type == PlotType.TOPICS:
            self.show_topics()
        elif self.current_plot_type == PlotType.TIME_COURSE:
            self.show_time_course()

    def show_gui(self) -> None:
        """
        Set up the GUI
        """
        self.gui = tkinter.Tk()
        # initial size of window
        self.gui.geometry("1500x1200")
        self.gui.wm_title("News Analysis")
        # initial checkbox value for date filtering (initially disabled)
        self.date_check = tkinter.IntVar(value=0)
        # initial checkbox values to enable/disable parties (initially all enabled)
        self.cdu_check = tkinter.IntVar(value=1)
        self.csu_check = tkinter.IntVar(value=1)
        self.spd_check = tkinter.IntVar(value=1)
        self.afd_check = tkinter.IntVar(value=1)
        self.gruene_check = tkinter.IntVar(value=1)
        self.linke_check = tkinter.IntVar(value=1)
        # initial checkbox values to enable/disable media (initially all enabled)
        self.tagesschau_check = tkinter.IntVar(value=1)
        self.taz_check = tkinter.IntVar(value=1)
        self.bild_check = tkinter.IntVar(value=1)

        # button to show sentiment filtered by party
        button_by_party = tkinter.Button(
            self.gui, text="Sentiment by Party", command=lambda: self.show_sentiment(by_party=True)
        )
        button_by_party.grid(row=0, column=0)

        # button to show sentiment filtered by media/outlet
        button_by_outlet = tkinter.Button(
            self.gui, text="Sentiment by Outlet", command=lambda: self.show_sentiment(by_party=False)
        )
        button_by_outlet.grid(row=0, column=1)

        # button to show topics of parties
        button_topic_party = tkinter.Button(
            self.gui, text="Show Topics of Parties", command=lambda: self.show_topics(by_party=True)
        )
        button_topic_party.grid(row=0, column=2)

        # button to show topics of media
        button_topic_media = tkinter.Button(
            self.gui, text="Show Topics of Media", command=lambda: self.show_topics(by_party=False)
        )
        button_topic_media.grid(row=0, column=3)

        # button to show time course
        button_time_course = tkinter.Button(self.gui, text="Show Time Course", command=self.show_time_course)
        button_time_course.grid(row=0, column=4)

        # checkbox anf text fields to filter dates
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
        self.entry_date_from.grid(row=1, column=2)

        label_date_to = tkinter.Label(self.gui, text=" To ")
        label_date_to.grid(row=1, column=3)
        self.entry_date_to = tkinter.Entry(self.gui, bd=5)
        self.entry_date_to.grid(row=1, column=4)

        # checkbox to enable/disable parties
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

        # checkbox to enable/disable media
        check_tagesschau = tkinter.Checkbutton(
            self.gui, text="Tagesschau", variable=self.tagesschau_check, onvalue=1, offvalue=0
        )
        check_tagesschau.grid(row=3, column=0)
        check_taz = tkinter.Checkbutton(self.gui, text="TAZ", variable=self.taz_check, onvalue=1, offvalue=0)
        check_taz.grid(row=3, column=1)
        check_bild = tkinter.Checkbutton(self.gui, text="Bild", variable=self.bild_check, onvalue=1, offvalue=0)
        check_bild.grid(row=3, column=2)

        update_button = self.next_button = tkinter.Button(
            self.gui, text="Update filter criteria", command=self.update_gui
        )
        update_button.grid(row=4, column=0, columnspan=5)

        # button to show next plot of currently available plots (in plots list)
        self.next_button = tkinter.Button(self.gui, text="Show next", command=lambda: self.show_diagram(increase=True))
        self.next_button["state"] = "disabled"
        self.next_button.grid(row=6, column=2)

        # button to previous next plot of currently available plots (in plots list)
        self.previous_button = tkinter.Button(
            self.gui, text="Show previous", command=lambda: self.show_diagram(increase=False)
        )
        self.previous_button["state"] = "disabled"
        self.previous_button.grid(row=6, column=0)

        self.help_button = tkinter.Button(self.gui, text="What does this graph show?", command=lambda: self.popupmsg())
        self.help_button["state"] = "disabled"
        self.help_button.grid(row=6, column=4)

        github_link = tkinter.Label(
            self.gui,
            text="Click here for more information about the project (GitHub repository)",
            fg="blue",
            cursor="hand2",
        )
        github_link.bind("<Button-1>", lambda e: self.open_browser("https://github.com/yakuhzi/news-analysis"))
        github_link.grid(row=7, column=0, columnspan=6)

        # "Hack" for displaying topics correctly, otherwise they sometimes appear in pie charts
        self.show_topics(by_party=True)
        self.clear_plots()
        self.current_plot_type = None

        # show GUI
        self.gui.mainloop()
