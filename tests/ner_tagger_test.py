import unittest

from src.ner_tagger import NERTagger


class NERTaggerTest(unittest.TestCase):
    def test_tagging(self):
        ner_tagger = NERTagger()
        text_to_tag = "Angela Merkel ist die deutsche Bundeskanzlerin und sie ist Mitglied der CDU."
        person_list, organization_list = ner_tagger.tag(text_to_tag)
        self.assertEqual(
            person_list,
            ["Angela Merkel"],
            "The person list should contain Angela Merkel",
        )
        self.assertEqual(
            organization_list, ["CDU"], "The organization list should contain CDU"
        )


if __name__ == "__main__":
    unittest.main()
