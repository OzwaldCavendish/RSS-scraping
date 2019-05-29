import unittest
from rsspump import RSSPump

test_feed = RSSPump("http://feeds.bbci.co.uk/news/rss.xml")


class RSSPumpTestCase(unittest.TestCase):
    """ Tests the initiation, connection, fetch and processing of the RSSPump class. """

    def test_RSSPump_stories(self):
        """ Check there's some actual processed stories present. """
        self.assertGreater(len(test_feed.get_stories()), 0)

    def test_RSSPump_entities(self):
        """ Check entity extraction got something. """
        self.assertGreater(len(test_feed.get_entities()), 0)

    def test_RSSPump_refresh(self):
        """ See if the refresh worked (reliant on indicator from module). """
        self.assertEqual(test_feed.refresh(), 'Refreshed feed data.')


if __name__ == '__main__':
    unittest.main()