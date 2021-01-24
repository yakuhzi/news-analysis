from typing import List

from labeled_lda import LldaModel


class TopicClassification:
    def __init__(self, text_data: List[List[str]]):
        self.text_data = text_data

        self.seed_social_affairs = [
            word.lower()
            for word in [
                "Arbeit",
                "Soziales",
                "Ausländerintegration",
                "Behinderte",
                "Drogen",
                "Ehrenamt",
                "Familie",
                "Frauen",
                "Generationen",
                "Gesundheit",
                "Gleichstellung",
                "Integration",
                "Jugend",
                "KITA",
                "Kindergarten",
                "Betreuung",
                "Verwahranstalt",
                "Mindestlohn",
                "Qualifikation",
                "Pflegeberuf",
                "Ausbildung",
                "Prostitution",
                "Senioren",
                "Wohlfahrt",
            ]
        ]

        self.seed_culture_education = [
            word.lower()
            for word in [
                "Kultus",
                "Bafög",
                "Studiengebühren",
                "Bildung",
                "Denkmalschutz",
                "Doping",
                "Ethik",
                "Forschung",
                "Frü̈hkindliche",
                "Bildung",
                "Gentechnik",
                "Juristenausbildung",
                "Kirche",
                "Kultur",
                "Kunst",
                "Medienpolitik",
                "Schule",
                "Sport",
                "Vorschuljahr",
                "Weiterbildung",
                "Wissenschaft",
            ]
        ]

        self.seed_agriculture = [
            word.lower()
            for word in [
                "Landwirtschaft",
                "Ernährung",
                "Fischerei",
                "Forsten",
                "Gentechnik",
                "Jagd",
                "Kleingarten",
                "ländliche",
                "Räume",
                "Tourismus",
                "Verbraucherschutz",
                "Weinbau",
            ]
        ]

        self.seed_finance = [word.lower() for word in ["Finanzen", "Steuern"]]

        self.seed_justice = [
            word.lower()
            for word in [
                "Justiz",
                "Abtreibung",
                "Asyl",
                "Bankgeheimnis",
                "Bürgerrechte",
                "Datenschutz",
                "Frauenhaus",
                "Frauenhandel",
                "Menschenhandel",
                "Gefängnis",
                "Strafvollzug",
                "Gewalt",
                "Kriminalität",
            ]
        ]

        self.seed_internal_affairs = [
            word.lower()
            for word in [
                "Inneres",
                "Auswanderungswesen",
                "Bezirksverwaltung",
                "Verwaltung",
                "Bürgerbegehren",
                "Bürokratieabbau",
                "Demokratie",
                "Einwanderung",
                "Katastrophenschutz",
            ]
        ]

        self.seed_environment = [
            word.lower()
            for word in [
                "Hochwasserschutz",
                "Lärmschutz",
                "Wirtschaft",
                "Verkehr",
                "Atomausstieg",
                "Außenhandel",
                "Bahn",
                "Energie",
                "Existenzgründung",
                "Hafen",
                "Infrastruktur",
                "Innovation",
                "Kredtwesen",
                "Banken",
                "Medienstandort",
                "Infrastruktur",
                "Staatskanzlei",
                "Ministerpräsident",
            ]
        ]

        self.seed_security = [
            word.lower()
            for word in [
                "Außenpolitik",
                "Sicherheitspolitik",
                "Entwicklungshilfe",
                "Militär",
                "Verteidigung",
                "Wehrdienst",
                "Wehrpflicht",
                "Zivildienst",
            ]
        ]

    def train_llda(self):
        labeled_documents = self._label_documents()
        print(labeled_documents)

        llda_model = LldaModel(labeled_documents=labeled_documents, alpha_vector=0.01)
        print(llda_model)

        while True:
            print("iteration %s sampling..." % (llda_model.iteration + 1))
            llda_model.training(10)

            print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
            print("delta beta: %s" % llda_model.delta_beta)

            if llda_model.is_convergent(method="beta", delta=2.5):
                break

        return llda_model

    def predict_llda(self, llda_model, document):
        return llda_model.inference(document=document, iteration=100, times=10)

    def _label_documents(self):
        documents = map(lambda text: " ".join(text), self.text_data)
        labeled_documents = map(lambda document: (document, self._get_seed_words(document)), documents)

        # return [("example example example example example "*100, ["example"]),
        #                      ("test arbeit llda model test llda model test llda model "*100, ["test", "llda_model"]),
        #                      ("example test example test example test example test "*100, ["example", "test"]),
        #                      ("good perfect good good perfect good good perfect good "*100, ["positive"]),
        #                      ("job arbeit corona arbeitsamt "*100, ["example"]),
        #                      ("bad bad down down bad bad down "*100, ["negative"])]

        return list(labeled_documents)

    def _get_seed_words(self, document):
        seed_words = []

        for word in document.split():
            if word in self.seed_social_affairs:
                self._append_seed_word(seed_words, "Social Affairs and Labour Market")
            elif word in self.seed_culture_education:
                self._append_seed_word(seed_words, "Culture and Education")
            elif word in self.seed_agriculture:
                self._append_seed_word(seed_words, "Agriculture")
            elif word in self.seed_finance:
                self._append_seed_word(seed_words, "Finance")
            elif word in self.seed_justice:
                self._append_seed_word(seed_words, "Justice")
            elif word in self.seed_internal_affairs:
                self._append_seed_word(seed_words, "Internal Affairs")
            elif word in self.seed_environment:
                self._append_seed_word(seed_words, "Environment and Regional Planning")
            elif word in self.seed_security:
                self._append_seed_word(seed_words, "Security and Foreign Affairs")

        # if len(seed_words) == 0:
        # return ["Common Topic"]

        return seed_words

    @staticmethod
    def _append_seed_word(seed_words, word):
        if word not in seed_words:
            seed_words.append(word)
