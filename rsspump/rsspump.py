""" Utility functions for scraping large numbers of RSS feeds, handles extracting info from multiple feeds, cleaning
    and returning that data in a DB ready schema/format for me.
    Add; url, fetch function, raw, processed (on call?) and separate timestamp querying.
    General principle; one pump per source!
    Can get source metadata from parser 'feed' """

import feedparser
import spacy
import numpy as np
from datetime import datetime

# Load the english language model for spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


class RSSPump:
    """ Instantiates a pump for a single RSS feed, connecting, downloading and parsing the text and entities from it."""
    def __init__(self, source_url, instantiate_run=False):
        self.source_url = source_url
        self.feed_object = feedparser.parse(self.source_url)
        self.corpus = self.get_stories()
        self.entities = self.get_entities()
        self.ignore_tags = ['CARDINAL', 'DATE', 'MONEY', 'ORDINAL', 'QUANTITY']

    def get_stories(self):
        """ Takes all articles and extracts the useful content and metadata in an SQL friendly shape."""

        try:
            if len(self.corpus) != 0:
                return self.corpus
        except:
            pass

        self.corpus = []
        for article in self.feed_object['entries']:
            try:
                self.corpus.append({"title": article['title'],  # News titles
                                    "summary": article['summary'].split("<")[0],  # payload (sans any HTML stuff)
                                    "date": article['published'],
                                    "link": article['links'][0]['href'],  # associated links
                                    "source_url": self.source_url,
                                    "retrieval_timestamp": str(datetime.now())})

            except KeyError as e:
                print("failed on ", article, e)
        return self.corpus

    def get_entities(self):
        """
        Uses SpaCy to extract all named entities.
        These named entities are stored with their pos as a string '<entity>:<pos_tag>' for convenience of DB storage.
        Do I need to create the new corpus here?  I don't know!
        """
        try:
            if len(self.entities) != 0:
                return self.entities
        except:
            pass

        self.entities = []
        for story in self.corpus:
            try:
                doc = nlp(story['title'] + " " + story['summary'])
                ents = ",".join([x.label_ + ":" + x.text for x in doc.ents if x.label_ not in self.ignore_tags])

                self.entities.append({'source_url':story['source_url'], 'title':story['title'], 'entities':ents})

            except:
                self.entities.append({'source_url': story['source_url'], 'title': story['title'], 'entities': "Failed"})

        return self.entities