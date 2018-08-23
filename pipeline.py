import os
import re
import string
import random
import time
import heapq
import wikipedia
import numpy
import nltk.tokenize
import nltk.stem.wordnet
import gensim
from sklearn import metrics
from sklearn import manifold


class Pipeline(object):
    """The full pipeline, proceeding from extracted Wikipedia text to derived models."""

    def __init__(self, verbose=True):
        """Initialize a Pipeline object."""
        start_time = time.time()
        self.verbose = verbose
        # Load in the raw corpus
        self.corpus = self.load_raw_corpus(path_to_corpus_files=PATH_TO_CORPUS_DIR_WITH_TSV_FILES)
        # Cull short articles
        self.cull_games_with_short_articles()
        # Cull any duplicates found in the corpus
        self.cull_duplicates()
        # Cull miscellaneous articles that were extracted in error
        self.cull_miscellaneous_erroneous_documents()
        # Sort corpus by pageid
        self.corpus.sort(key=lambda g: int(g.pageid))
        # Attribute platforms to the games in the corpus
        self.attribute_platforms_to_games()
        # Build a dictionary mapping multiword game titles to their tokenizations
        self.multiword_game_titles = self.build_multiword_game_titles_dictionary()
        # Build a dictionary mapping multiword platform names to their tokenizations
        self.multiword_game_platforms = self.build_multiword_game_platforms_dictionary()
        # Build a multiword-phrase tokenizer
        self.multiword_phrase_tokenizer = self.build_multiword_phrase_tokenizer()
        # Precompute lemmatizations for each word that appears in the corpus
        self.lemmatizations = self.precompute_lemmatizations()
        # Use these resources to preprocess the corpus
        self.preprocess_corpus()
        # Isolate and remove all useless terms (ones that occur in only a single document)
        useless_terms = self.isolate_useless_terms()
        self.remove_useless_terms(useless_terms=useless_terms)
        # Build term-id dictionary
        self.term_id_dictionary = self.build_term_id_dictionary()
        # Build and save serialized corpus
        self.serialize_corpus()
        # Derive tf-idf model
        self.tf_idf_model = self.derive_tf_idf_model()
        # Derive LSA model
        self.lsa_index = None  # Gets set by self.derive_lsa_model()
        self.derive_lsa_model()
        # Update GameNet
        self.update_gamenet()
        # Update GameSpace
        self.update_gamespace(force_distance_array=True)
        # Delete any temporary files that were created to support processing
        self.delete_temporary_files()
        if self.verbose:
            print "Done! The entire process took {n} seconds.".format(n=int(time.time()-start_time))

    def load_raw_corpus(self, path_to_corpus_files):
        """Return a list of Game objects."""
        if self.verbose:
            print "Loading in raw corpus..."
        corpus = []
        # Read in each TSV file in the corpus directory
        for tsv_file_name in os.listdir(path_to_corpus_files):
            # Create a file object
            path_to_this_tsv_file = '{dir}/{file}'.format(dir=path_to_corpus_files, file=tsv_file_name)
            tsv_file = open(path_to_this_tsv_file, 'r')
            # Read in the lines and throw out the header
            lines = tsv_file.readlines()[1:]
            # Slurp in each line and build a corresponding Game object (here, we ignore the field 'revision_id')
            for line in lines:
                title, pageid, _, year, intro_text_html, raw_text, categories_str = line.split('\t')
                raw_text = raw_text.decode('utf-8')
                game_object = Game(
                    pageid=pageid, title=title, year=year, intro_text_html=intro_text_html,
                    raw_text=raw_text, categories_str=categories_str
                )
                corpus.append(game_object)
        return corpus

    def cull_games_with_short_articles(self):
        """Remove games from the corpus whose articles are too short."""

        N_REMOVED = 0

        if self.verbose:
            print "Removing short articles..."
        for game in list(self.corpus):
            if len(game.raw_text.split()) < THRESHOLD_FOR_DOCUMENT_LENGTH:
                N_REMOVED += 1
                self.corpus.remove(game)

        print "REMOVED {} SHORT ARTICLES".format(N_REMOVED)

    def cull_duplicates(self):
        """Isolate duplicate games in the corpus and remove all but one of each."""

        N_REMOVED = 0

        if self.verbose:
            print "Removing duplicate games..."
        # Remove documents that are duplicates of other documents in the corpus; this appears to occur
        # when two games that are in the same series, or related in some other way, are not notable
        # enough to each have their own Wikipedia articles and so share a single article
        duplicate_sets = []
        already_detected = set()
        for i in xrange(len(self.corpus)):
            game = self.corpus[i]
            # Print out a progress update
            if self.verbose:
                if i % 1000 == 0:
                    print '\t{i}/{n}'.format(i=i, n=len(self.corpus))
            if game not in already_detected:
                set_of_duplicates = {game}
                for other_game in self.corpus:
                    if other_game is not game:
                        if other_game.pageid == game.pageid:
                            set_of_duplicates.add(other_game)
                if set_of_duplicates > 1:
                    already_detected |= set_of_duplicates
                    duplicate_sets.append(set_of_duplicates)
        # Pick an archetype for each set, and record data about any alternative titles and years
        for duplicate_set in duplicate_sets:
            # Pick the game with the earliest year of release to be the archetype (letting Python
            # resolve any ties arbitrarily)
            archetype = min(duplicate_set, key=lambda g: g.year)
            for dupe in duplicate_set:
                if dupe is not archetype:
                    # Set its data as alternative data on the archetype
                    if dupe.title != archetype.title:
                        archetype.alternate_titles.append(dupe.title)
                    if dupe.year != archetype.year:
                        archetype.alternate_years.append(dupe.year)
                    # Remove the duplicate from the corpus
                    self.corpus.remove(dupe)
                    N_REMOVED += 1

        print "REMOVED {} DUPES".format(N_REMOVED)

    def cull_miscellaneous_erroneous_documents(self):
        """Cull documents that were erroneously extracted (for various reasons)."""
        # Remove Wikipedia user pages
        for game in list(self.corpus):
            if 'User:' in game.title:
                self.corpus.remove(game)

    def attribute_platforms_to_games(self):
        """Attribute platforms of release to each game in the corpus."""
        if self.verbose:
            print "Attributing platforms to games..."
        # Parse our handcrafted resources related to videogame platforms -- we'll need to
        # use them shortly
        pageid_to_platforms, alternate_to_canonical_platform_name, platforms_ranked_by_salience = (
            self.parse_platform_resource_files()
        )
        # Attribute platforms
        self.attribute_all_platforms(pageid_to_platforms=pageid_to_platforms)
        # Attribute primary platforms
        self.attribute_primary_platforms(
            alternate_to_canonical_platform_name=alternate_to_canonical_platform_name,
            platforms_ranked_by_salience=platforms_ranked_by_salience
        )
        # Update the file 'pageid_to_platforms.tsv' to include any new information that we've computed; this
        # way, we don't have to recompute all this new information again (this computation happens to be costly)
        self.update_platform_resource_files()

    @staticmethod
    def parse_platform_resource_files():
        """Parse our handcrafted resources that relate to videogame platforms."""
        # Parse a TSV file that stores precomputed information about which platforms (primary
        # or otherwise) each game in the corpus has been released on; later, we'll update this
        # for any new information extracted during this pipeline session
        f = open('pageid_to_platforms.tsv', 'r')
        pageid_to_platforms = {}
        for pageid, primary_platform, platforms in [line.lower().strip('\n').split('\t') for line in f.readlines()][1:]:
            pageid_to_platforms[pageid] = [primary_platform, platforms.split(',')]
        # The file maps variant names for each known platform to our canonical name for that
        # platform; here, 'canonical' is optimized to support the success of autogenerated
        # queries, i.e., they are the names I thought would work best in YouTube queries
        f = open('platform_canonical_and_alternate_names.tsv', 'r')
        entries = [line.strip('\n').split('\t') for line in f.readlines()]
        alternate_to_canonical_platform_name = {}
        for canonical_name, alternates in entries:
            for alternate in alternates.split(','):
                if alternate:
                    alternate_to_canonical_platform_name[alternate] = canonical_name
        # This file ranks all our known videogame platforms according to their salience; if a
        # game in the corpus has multiple platforms associated with it, the one that ranks most
        # highly in this list will be selected as that game's primary platform (which in turn will
        # be used in the web apps to autogenerate image and video queries)
        f = open('platforms_ranked_by_salience.txt', 'r')
        platforms_ranked_by_salience = [line.strip('\n') for line in f.readlines()]
        return pageid_to_platforms, alternate_to_canonical_platform_name, platforms_ranked_by_salience

    def attribute_all_platforms(self, pageid_to_platforms):
        """Attribute to each game a list of all known platforms that it was released on."""
        for game in self.corpus:
            if game.pageid in pageid_to_platforms:
                game.primary_platform, game.platforms = pageid_to_platforms[game.pageid]
            else:  # Extract platforms from the HTML of the game's Wikipedia article
                print self.corpus.index(game)
                try:
                    wikipedia_page = wikipedia.page(pageid=game.pageid)
                    article_html = wikipedia_page.html()
                    if '"Computing platform">Platform(s)' in article_html:
                        game.platforms = self.extract_platforms_from_article_infobox(article_html=article_html)
                    else:
                        article_summary = wikipedia_page.summary.encode('utf-8')
                        game.platforms = self.extract_platforms_from_article_summary(article_summary=article_summary)
                except wikipedia.exceptions.PageError:  # The page has been removed from Wikipedia
                    game.platforms = []
                    game.primary_platform = None

    @staticmethod
    def extract_platforms_from_article_infobox(article_html):
        """Extract a game's platforms by parsing the infobox of its Wikipedia-article HTML."""
        release_platforms = []
        # Parse out the relevant portions of infobox
        start_of_span_pertaining_to_platforms = article_html.index('"Computing platform">Platform(s)')
        span_pertaining_to_platforms = article_html[start_of_span_pertaining_to_platforms:]
        end_of_span_pertaining_to_platforms = span_pertaining_to_platforms.index('</span>')
        span_pertaining_to_platforms = span_pertaining_to_platforms[:end_of_span_pertaining_to_platforms]
        while 'title="' in span_pertaining_to_platforms:
            # Consume the file for its next listed platform
            start_of_segment_naming_next_platform = span_pertaining_to_platforms.index('title="') + 7
            segment_naming_next_platform = span_pertaining_to_platforms[start_of_segment_naming_next_platform:]
            end_of_segment_naming_next_platform = segment_naming_next_platform.index('"')
            name_of_next_platform = segment_naming_next_platform[:end_of_segment_naming_next_platform]
            release_platforms.append(name_of_next_platform.lower())
            # Remove portion of the span pertaining to platforms that lists the just-extracted platform
            try:
                new_start_of_span = (
                    span_pertaining_to_platforms.index('title="{}"'.format(name_of_next_platform)) +
                    len('title="{}"'.format(name_of_next_platform))
                )
            except ValueError:
                new_start_of_span = span_pertaining_to_platforms.index('title="') + 8
            span_pertaining_to_platforms = span_pertaining_to_platforms[new_start_of_span:]
        for i in xrange(len(release_platforms)):
            if release_platforms[i] == 'Personal computer':
                release_platforms[i] = 'PC'
        return release_platforms

    @staticmethod
    def extract_platforms_from_article_summary(article_summary):
        """Extract a game's platforms by parsing the infobox of its Wikipedia-article HTML."""
        known_platforms = [_.strip('\n').lower() for _ in open('platforms.txt')]
        release_platforms = []
        try:
            first_sentence_of_wiki_summary = article_summary.split('.')[0]
            for platform in known_platforms:
                if ' ' + platform.lower() in first_sentence_of_wiki_summary.lower():
                    release_platforms.append(platform.lower())
            if not release_platforms:
                for platform in known_platforms:
                    if ' ' + platform.lower() in article_summary.lower():
                        if platform not in release_platforms:
                            release_platforms.append(platform.lower())
        except ValueError:
            pass
        return release_platforms

    def attribute_primary_platforms(self, alternate_to_canonical_platform_name, platforms_ranked_by_salience):
        """Attribute to each game it's primary platform."""
        for game in (g for g in self.corpus if not g.primary_platform):
            if not game.platforms:
                game.primary_platform = ''
            else:
                candidates_for_primary_platform = set()
                for platform in game.platforms:
                    if platform in alternate_to_canonical_platform_name:
                        candidates_for_primary_platform.add(alternate_to_canonical_platform_name[platform])
                    else:
                        candidates_for_primary_platform.add(platform)
                known_candidates = filter(lambda p: p in platforms_ranked_by_salience, candidates_for_primary_platform)
                if known_candidates:
                    game.primary_platform = min(known_candidates, key=lambda p: platforms_ranked_by_salience.index(p))
                else:
                    game.primary_platform = game.platforms[0]  # Just pick the one that was listed first in the infobox

    def update_platform_resource_files(self):
        """Update resource files containing information about videogame platforms (for new extracted information)."""
        # First, we need to remove any weird characters that could cause encoding issues during file writing
        for game in self.corpus:
            game.primary_platform = filter(lambda char: char in string.printable, game.primary_platform)
            for i in xrange(len(game.platforms)):
                game.platforms[i] = filter(lambda char: char in string.printable, game.platforms[i])
        # Now, write the file
        f = open('pageid_to_platforms.tsv', 'w')
        f.write('pageid\tprimary_platform\tplatforms\n')
        for game in self.corpus:
            line = '{pageid}\t{primary_platform}\t{platforms_str}\n'.format(
                pageid=game.pageid,
                primary_platform=game.primary_platform,
                platforms_str=','.join(game.platforms)
            )
            f.write(line)
        f.close()

    def build_multiword_game_titles_dictionary(self):
        """Return a dictionary mapping multiword game titles to their tokenizations."""
        if self.verbose:
            print "Building dictionary for multiword game titles..."
        titles = [game.title.lower() for game in self.corpus]
        multiword_titles = [title for title in titles if len(title.split()) > 1]
        multiword_titles_dictionary = {}
        for multiword_title in multiword_titles:
            cleaned_multiword_title = self.remove_punctuation(raw_text=multiword_title)
            multiword_titles_dictionary[multiword_title] = '_'.join(cleaned_multiword_title.split())
        return multiword_titles_dictionary

    def build_multiword_game_platforms_dictionary(self):
        """Return a dictionary mapping multiword platform names to their tokenizations."""
        if self.verbose:
            print "Building dictionary for multiword game platforms..."
        platform_names = [_.strip('\n').lower() for _ in open('platforms.txt')]
        multiword_platform_names = [title for title in platform_names if len(title.split()) > 1]
        multiword_platforms_dictionary = {}
        for multiword_platform_name in multiword_platform_names:
            cleaned_multiword_platform_name = self.remove_punctuation(raw_text=multiword_platform_name)
            multiword_platforms_dictionary[multiword_platform_name] = '_'.join(cleaned_multiword_platform_name.split())
        return multiword_platforms_dictionary

    def build_multiword_phrase_tokenizer(self):
        """Return a multiword-phrase tokenizer."""
        if self.verbose:
            print "Building multiword phrase tokenizer..."
        # Split the corpus into a set of sentences, each with its punctuation removed
        all_cleaned_sentences = []
        for game in self.corpus:
            raw_text = game.raw_text
            sentences = nltk.tokenize.sent_tokenize(raw_text)
            for sentence in sentences:
                sentence = self.remove_punctuation(raw_text=sentence)
                sentence = sentence.split()
                all_cleaned_sentences.append(sentence)
        # Train a bigram tokenizer using gensim
        bigram_tokenizer = gensim.models.Phrases(all_cleaned_sentences)
        # Bootstrap that to train a trigram tokenizer
        trigram_tokenizer = gensim.models.Phrases(bigram_tokenizer[all_cleaned_sentences])
        # Note: could always move on to tetragrams or further
        return trigram_tokenizer

    def precompute_lemmatizations(self):
        """Return a dictionary mapping all words that appear in the corpus to their lemmatizations."""
        if self.verbose:
            print "Precomputing lemmatizations..."
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        lemmatizations = {}
        for game in self.corpus:
            text = game.raw_text
            tokens = text.split()
            for word in tokens:
                if word not in lemmatizations:
                    lemmatizations[word] = lemmatizer.lemmatize(word)
        return lemmatizations

    def preprocess_corpus(self):
        """Preprocess the raw text of each entry in the corpus."""
        if self.verbose:
            print "Preprocessing corpus..."
        for i in xrange(len(self.corpus)):
            game = self.corpus[i]
            # Print out a progress update
            if self.verbose:
                if i % 1000 == 0:
                    print '\t{i}/{n}'.format(i=i, n=len(self.corpus))
            # Remove punctuation
            preprocessed_text = self.remove_punctuation(game.raw_text)
            # Tokenize multiword game titles
            titles_sorted_by_number_of_words = (
                sorted(self.multiword_game_titles.keys(), key=lambda t: len(t.split()), reverse=True)
            )
            preprocessed_text = self.tokenize_multiword_game_titles(
                text=preprocessed_text,
                multiword_game_titles=self.multiword_game_titles,
                titles_sorted_by_number_of_words=titles_sorted_by_number_of_words
            )
            # Tokenize multiword platform names
            platform_names_sorted_by_number_of_words = (
                sorted(self.multiword_game_platforms.keys(), key=lambda p: len(p.split()), reverse=True)
            )
            preprocessed_text = self.tokenize_multiword_platforms(
                text=preprocessed_text,
                multiword_game_platforms=self.multiword_game_platforms,
                platform_names_sorted_by_number_of_words=platform_names_sorted_by_number_of_words
            )
            # Tokenize multiword phrases
            preprocessed_text = ' '.join(
                self.multiword_phrase_tokenizer[self.multiword_phrase_tokenizer[preprocessed_text.split()]]
            )
            # Remove stopwords
            preprocessed_text = self.remove_stopwords(text=preprocessed_text)
            # Lemmatize text
            preprocessed_text = self.lemmatize(text=preprocessed_text, lemmatizations=self.lemmatizations)
            # Remove stopwords again
            preprocessed_text = self.remove_stopwords(text=preprocessed_text)
            # Save the preprocessed text as the '.text' attribute of the Game object at hand
            game.text = preprocessed_text

    @staticmethod
    def remove_punctuation(raw_text):
        """Clean raw text by removing punctuation."""
        # Convert to lowercase
        raw_text = raw_text.lower()
        # Remove Wikipedia-style citations, e.g., '[3]'
        raw_text = re.sub(r'\[[^)]*\]', '', raw_text)
        # Replace punctuation and other symbols with whitespace
        for symbol in [
            '.', '!', '?', ';', ',', '[', ']', "'", '"', ':',
            '&', '(', ')', '\\', '/', '*', '$', '^', '~', '+',
            '=', '{', '}', '`', '|', '#', '_', '-', '\r', '\n',
        ]:
            raw_text = raw_text.replace(symbol, ' ')
        raw_text = raw_text.replace("'", '')  # Due to possessive constructions, we want this to work differently
        # Remove redundant whitespace
        raw_text = ' '.join(raw_text.split())
        return raw_text

    @staticmethod
    def tokenize_multiword_game_titles(text, multiword_game_titles, titles_sorted_by_number_of_words):
        """Tokenize occurrences of multiword game titles."""
        # Prepend and append single whitespace characters to catch title appearances at the beginning
        # and end of an article's text
        tokenized_text = ' ' + text + ' '
        # Tokenize multiword game titles
        for title in titles_sorted_by_number_of_words:
            tokenized_title = multiword_game_titles[title]
            try:
                while (' ' + title + ' ').decode('utf-8') in tokenized_text:
                    tokenized_text = tokenized_text.replace(
                        (' ' + title + ' ').decode('utf-8'),
                        ' ' + tokenized_title + ' '
                    )
            except UnicodeDecodeError:
                # JOR: Spent several hours trying to figure this out; just going to say screw it
                # on these titles
                pass
        return tokenized_text

    @staticmethod
    def tokenize_multiword_platforms(text, multiword_game_platforms, platform_names_sorted_by_number_of_words):
        """Tokenize occurrences of multiword game titles."""
        # Prepend and append single whitespace characters to catch appearances of platform names
        # at the beginning and end of an article's text
        tokenized_text = ' ' + text + ' '
        # Tokenize multiword game platforms
        for platform_name in platform_names_sorted_by_number_of_words:
            tokenized_platform_name = multiword_game_platforms[platform_name]
            while (' ' + platform_name + ' ').decode('utf-8') in tokenized_text:
                tokenized_text = tokenized_text.replace(
                    (' ' + platform_name + ' ').decode('utf-8'),
                    (' ' + tokenized_platform_name + ' ').decode('utf-8'),
                )
        return tokenized_text

    @staticmethod
    def remove_stopwords(text):
        """Remove all stopwords from the given text."""
        stopwords = [_.strip('\n') for _ in open('stopwords.txt')]
        tokens = [token.lower() for token in text.split()]
        for i in xrange(len(tokens)):
            if tokens[i] in stopwords:
                tokens[i] = ''
            elif len(tokens[i]) == 1:  # Also remove all single-character tokens
                tokens[i] = ''
        text_without_stopwords = ' '.join([token for token in tokens if token])
        return text_without_stopwords

    @staticmethod
    def lemmatize(text, lemmatizations):
        """Lemmatize all words in the given text."""
        lemmatized_text = text
        # Do five passes, just to be safe
        for i in xrange(5):
            tokens = lemmatized_text.split()
            for j in xrange(len(tokens)):
                try:
                    tokens[j] = lemmatizations[tokens[j]]
                except KeyError:
                    # During last pass, words were turned into their lemmas, which don't
                    # have entries in lemmatizations
                    pass
            lemmatized_text = ' '.join(tokens)
        return lemmatized_text

    def isolate_useless_terms(self):
        """Isolate all terms that occur in only one document in the corpus."""
        if self.verbose:
            print "Isolating useless terms..."
        # First, survey all terms that appear anywhere in the corpus
        terms = set()
        for game in self.corpus:
            for token in game.text.split():
                terms.add(token)
        terms = list(terms)
        # Determine which terms occur in only one document
        documents = [set(game.text.split()) for game in self.corpus]
        terms_occurring_in_only_one_document = set()
        for i in xrange(len(terms)):
            term = terms[i]
            # Print out a progress update
            if self.verbose:
                if i % 1000 == 0:
                    print "\t{i}/{n}".format(i=i, n=len(terms))
            n_documents_term_appears_in = 0
            for document in documents:
                if term in document:
                    n_documents_term_appears_in += 1
                    if n_documents_term_appears_in > 1:
                        break
            if n_documents_term_appears_in < 2:
                terms_occurring_in_only_one_document.add(term)
        return terms_occurring_in_only_one_document

    def remove_useless_terms(self, useless_terms):
        """Remove all terms that occur in only a single document."""
        if self.verbose:
            print "Removing useless terms..."
        for i in xrange(len(self.corpus)):
            game = self.corpus[i]
            # Print out a progress update
            if self.verbose:
                if i % 1000 == 0:
                    print '\t{i}/{n}'.format(i=i, n=len(self.corpus))
            # Remove useless terms
            document = [token for token in game.text.split() if token not in useless_terms]
            game.text = ' '.join(document)

    def build_term_id_dictionary(self):
        """Build a term-id dictionary."""
        # Collect corpus documents
        documents = []
        for game in self.corpus:
            documents.append(game.text.split())
        # Build term-ID dictionary
        term_id_dictionary = gensim.corpora.Dictionary(documents)
        # Update GameNet by writing the dictionary to its static directory
        term_id_dictionary.save('{dir}/ontology-id2term.dict'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO))
        return term_id_dictionary

    def serialize_corpus(self):
        """Serialize the corpus to produce an MM file."""
        if self.verbose:
            print 'Serializing corpus...'
        # Collect corpus documents
        documents = []
        for game in self.corpus:
            documents.append(game.text.split())
        # Serialize corpus to MM format
        serialized_corpus = [self.term_id_dictionary.doc2bow(document) for document in documents]
        gensim.corpora.MmCorpus.serialize('TEMP_DEL_serialized_corpus.mm', serialized_corpus)

    def derive_tf_idf_model(self):
        """Derive a tf-idf model from the serialized corpus."""
        if self.verbose:
            print 'Deriving tf-idf model...'
        # Load serialized corpus
        serialized_corpus = gensim.corpora.MmCorpus('TEMP_DEL_serialized_corpus.mm')
        # Train a tf-idf model on the corpus so that any document can be converted from its
        # serialized representation to a tf-idf representation
        tf_idf_transformer = gensim.models.TfidfModel(serialized_corpus)
        # Build tf-idf representations for every document in the corpus
        corpus_tf_idf_model = tf_idf_transformer[serialized_corpus]
        # Update GameNet by writing the tf-idf model file to its static directory
        corpus_tf_idf_model.save('{dir}/ontology-tfidf_model'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO))
        return corpus_tf_idf_model

    def derive_lsa_model(self):
        """Derive a latent semantic analysis model."""
        if self.verbose:
            print "Deriving LSA model..."
        # Derive LSA model
        lsa_model = gensim.models.LsiModel(corpus=self.tf_idf_model, id2word=self.term_id_dictionary, num_topics=K)
        # Update GameNet by writing the LSA model file to its static directory
        lsa_model.save('{dir}/ontology-model_{k}.lsi'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO, k=K))
        # Attribute LSA vectors to each of the games in the corpus
        serialized_corpus = gensim.corpora.MmCorpus('TEMP_DEL_serialized_corpus.mm')
        tf_idf_transformer = gensim.models.TfidfModel(serialized_corpus)
        for i in xrange(len(self.corpus)):
            game_with_this_index = self.corpus[i]
            frequency_count_vector_for_that_game = serialized_corpus[i]
            tf_idf_vector = tf_idf_transformer[frequency_count_vector_for_that_game]
            document_lsa_vector = lsa_model[tf_idf_vector]
            # The LSA vector will be a list of tuples, each of the form, e.g., '(179, 0.0034549460187201799)',
            # where 179 is the dimension number and the second element in the tuple is the score along that
            # dimension; here, want to exclude the useless first dimension, but we also need the dimension
            # numbers to start at 0 and proceed with no gaps (gensim expects in build_pairwise_distance_matrix);
            # let's create a list comprehension accordingly, and then attribute the result to the game at hand
            game_with_this_index.lsa_vector = [(score[0]-1, score[1]) for score in document_lsa_vector[1:]]
        # Also derive the LSA model index, which is what we use for efficient computation of
        # document-document similarity
        self.lsa_index = gensim.similarities.docsim.Similarity(
            output_prefix='TEMP_DEL_', corpus=[game.lsa_vector for game in self.corpus], num_features=207,
            num_best=len(self.corpus)
        )
        # Write out that index file to the GameNet directory, as well (I believe it's used by GameSage)
        self.lsa_index.save('{dir}/ontology-model_{k}.index'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO, k=K))

    def update_gamenet(self):
        """Update GameNet by writing out its various files."""
        # First, let's save the multiword-phrase tokenizer that we built earlier; ontology
        # GameSage will now use this during preprocessing
        self.multiword_phrase_tokenizer.save(
            '{dir}/ontology-multiword_phrase_tokenizer'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO)
        )
        # Next, we need to collect each game's fifty most- and least-related games
        self.attribute_most_and_least_related_games()
        # Now, write out an updated TSV file with game metadata
        self.write_gamenet_games_metadata_tsv()
        # Next, write out the metadata file containing each game's LSA vector
        self.write_gamenet_lsa_vectors_tsv()
        # Lastly, write out the JSON file containing game metadata that's needed
        # for autocompletion in the GameNet search bar
        self.write_gamenet_autocompletion_json()

    def attribute_most_and_least_related_games(self):
        """Attribute to each game its fifty most- and least-related games."""
        # Iterate over the index to attribute LSA scores between that game and its most-
        # and least-related games
        if self.verbose:
            print "Attributing fifty most- and least-related games..."
        for i in xrange(len(self.corpus)):
            game = self.corpus[i]
            # Print out a progress update
            if self.verbose:
                if i % 1000 == 0:
                    print '\t{i}/{n}'.format(i=i, n=len(self.corpus))
            # Determine LSA scores for all games relative to this game
            lsa_scores_for_all_games_relative_to_this_game = self.lsa_index[game.lsa_vector]
            game.lsa_scores = lsa_scores_for_all_games_relative_to_this_game
            # Attribute the 50 most related and unrelated games, replacing corpus indices with the games'
            # pageids (since these persist, allowing for, e.g., user bookmarking of GameNet entries); here,
            # we want to be sure to remove the game itself from the listing of its most related games
            most_related = heapq.nlargest(
                51, lsa_scores_for_all_games_relative_to_this_game, key=lambda game_id_and_score: game_id_and_score[1]
            )
            game.related_games = [g for g in most_related if g[0] != i]  # Exclude the game itself
            game.unrelated_games = heapq.nsmallest(
                50, lsa_scores_for_all_games_relative_to_this_game, key=lambda game_id_and_score: game_id_and_score[1]
            )

    def write_gamenet_games_metadata_tsv(self):
        """Update GameNet by rewriting its TSV file with game metadata."""
        if self.verbose:
            print "Writing GameNet metadata TSV..."
        f = open('{dir}/games_metadata-ontology.tsv'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO), 'w')
        for game in self.corpus:
            # Do it this dumb way to avoid dreaded unicode errors
            f.write(game.pageid + '\t')
            f.write(game.title + '\t')
            f.write(str(game.year) + '\t')
            f.write(game.primary_platform if game.primary_platform else '' + '\t')
            f.write(game.wikipedia_url + '\t')
            f.write(game.wikipedia_summary + '\t')
            # Separate the fifty related scores by commas, with '&' symbols separating IDs and scores
            for game_id, score in game.related_games[:-1]:
                f.write('{game_id}&{score},'.format(game_id=game_id, score=score))
            # No trailing comma for the last one
            f.write('{game_id}&{score}'.format(game_id=game.related_games[-1][0], score=game.related_games[-1][1]))
            f.write('\t')
            # Do the same for unrelated games
            for game_id, score in game.unrelated_games[:-1]:
                f.write('{game_id}&{score},'.format(game_id=game_id, score=score))
            f.write('{game_id}&{score}'.format(game_id=game.unrelated_games[-1][0], score=game.unrelated_games[-1][1]))
            f.write('\t')
            # Write trailing carriage for all except the last element in the TSV
            if self.corpus.index(game) != len(self.corpus)-1:
                f.write('\n')
        f.close()

    def write_gamenet_lsa_vectors_tsv(self):
        """Update GameNet by rewriting its TSV file with game LSA vectors."""
        f = open('{dir}/game_lsa_vectors-ontology.tsv'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO), 'w')
        for game in self.corpus:
            line = '{game_id}\t{title}\t{year}\t{lsa_vector_str}\n'.format(
                game_id=game.pageid,
                title=game.title,
                year=game.year,
                lsa_vector_str=','.join(str(score[1]) for score in game.lsa_vector)
            )
            f.write(line)
        f.close()

    def write_gamenet_autocompletion_json(self):
        """Update GameNet by rewriting its JSON file that is used to support autocompletion."""
        f = open('{dir}/gamesDataForAutocomplete-ontology.json'.format(dir=PATH_TO_WRITE_GAMENET_FILES_TO), 'w')
        f.write('var games_list = [\n')
        for game in self.corpus[:-1]:
            line = '  {{"label" : "{title}", "{game_id}" : "0"}},\n'.format(
                title=game.title,
                game_id=game.pageid
            )
            f.write(line)
        # JSON can't handle a trailing comma
        last_line = '  {{"label" : "{title}", "{game_id}" : "0"}}'.format(
            title=self.corpus[-1].title,
            game_id=self.corpus[-1].pageid
        )
        f.write(last_line)
        f.write('\n]')
        f.close()

    def update_gamespace(self, force_distance_array=False):
        """Update GameSpace by writing out its various files."""
        # Derive 3D coordinates for the games (for use in GameSpace)
        self.derive_3d_coordinates(force_distance_array=False)
        # Write out GameSpace JSON files with 3D coordinates and metadata
        self.write_gamespace_metadata_and_coordinates_jsons()
        if force_distance_array:
            # Derive 3D coordinates for the games (for use in GameSpace)
            self.derive_3d_coordinates(force_distance_array=True)
            # Write out GameSpace JSON files with 3D coordinates and metadata
            self.write_gamespace_metadata_and_coordinates_jsons()

    def derive_3d_coordinates(self, force_distance_array=False):
        """Derive three-dimensional coordinates for the games, using multidimensional scaling, and attribute them."""
        if self.verbose:
            print "Deriving 3D coordinates..."
        # Prepare LLE manifold
        three_dimensional_representation = manifold.LocallyLinearEmbedding(
            n_neighbors=15, n_components=3, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=10000,
            method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=1
        )
        if force_distance_array:
            # Build LSA feature matrix
            matrix_to_fit_to = self.build_pairwise_cosine_matrix()
        else:
            matrix_to_fit_to = self.build_lsa_feature_matrix()
        # Derive 3D coordinates by fitting the LSA feature matrix to the manifold
        results = three_dimensional_representation.fit(matrix_to_fit_to)
        all_game_3d_coordinates = results.embedding_
        # Attribute the derived 3D coordinates to the games
        for i in xrange(len(all_game_3d_coordinates)):
            game = self.corpus[i]
            game.three_dimensional_coordinates = all_game_3d_coordinates[i]

    def build_lsa_feature_matrix(self):
        """Build a matrix where each column is a game's LSA vector."""
        if self.verbose:
            print "Building LSA feature matrix..."
        # Here, we don't need the dimension numbers in the game's LSA vectors, which were required
        # by gensim in build_pairwise_distance_matrix()
        lsa_feature_matrix = numpy.zeros(shape=(len(self.corpus), len(self.corpus[0].lsa_vector)))
        for i in xrange(len(self.corpus)):
            game = self.corpus[i]
            lsa_vector = [score[1] for score in game.lsa_vector]
            lsa_feature_matrix[i] = lsa_vector
        return lsa_feature_matrix

    def build_pairwise_cosine_matrix(self):
        """Build a matrix containing pairwise cosine scores for all the games in the corpus."""
        if self.verbose:
            print "Building pairwise cosine matrix..."
        feature_matrix = numpy.zeros(shape=(len(self.corpus), len(self.corpus[0].lsa_vector)))
        for i in xrange(len(self.corpus)):
            game = self.corpus[i]
            lsa_vector = [score[1] for score in game.lsa_vector]
            feature_matrix[i] = lsa_vector
        pairwise_cosine_matrix = metrics.pairwise.pairwise_distances(feature_matrix, metric='cosine')
        # Normalize the matrix by dividing all cell values by the largest value in the matrix
        largest_cosine = numpy.amax(pairwise_cosine_matrix)
        pairwise_cosine_matrix /= largest_cosine
        return pairwise_cosine_matrix

    def write_gamespace_metadata_and_coordinates_jsons(self):
        """Update GameSpace by rewriting its metadata and coordinates JSON files."""
        if self.verbose:
            print "Writing GameSpace metadata JSONs..."
        # We'll be breaking the corpus into chunks of <= 1000 games, and writing out
        # a single JSON for each chunk
        number_of_files_needed = int(len(self.corpus)/1000) + 1
        for i in xrange(1, number_of_files_needed+1):
            # Generate a random string of characters to be included in the title (this is done
            # to combat browser caching, which we need to worry about once new games are
            # automatically added into the model periodically)
            random_str = ''.join(random.choice(string.letters+string.digits) for _ in range(6))
            f = open('{dir}/games{random_str}.json'.format(
                dir=PATH_TO_WRITE_GAMESPACE_FILES_TO, random_str=random_str
            ), 'w')
            f.write('[\n')
            batch = self.corpus[(i-1)*1000:i*1000]
            for game in batch:
                game_3d_coords = '{coords_str}'.format(
                    coords_str=','.join([str(c) for c in game.three_dimensional_coordinates])
                )
                title = game.title
                if '\\' in title:
                    title = title.replace('\\', '\\\\')
                if '"' in title:  # Not worth dealing with; Spy Fox includes weird double-double quotes
                    title = title.replace('"', '')
                # Write out this dumb way to avoid dreaded unicode errors
                f.write('  {"id" : ' + str(game.pageid))
                f.write(', "title" : "')
                f.write(title)
                year = str(game.year) if game.year != 'upcoming' else '"upcoming"'
                f.write('", "year" : ' + year)
                f.write(', "platform" : "')
                f.write(game.primary_platform)
                f.write('", "wiki_url" : "' + game.wikipedia_url)
                f.write('", "wiki_length" : "N/A"')  # We aren't using this currently, so not worth calculating
                corpus_index = self.corpus.index(game)
                if corpus_index == (i * 1000) - 1 or corpus_index == len(self.corpus) - 1:
                    # Omit trailing comma from last object in the JSON
                    f.write(', "coords" : [' + game_3d_coords + ']}\n')
                else:
                    f.write(', "coords" : [' + game_3d_coords + ']},\n')
            f.write(']')
            f.close()

    @staticmethod
    def delete_temporary_files():
        """Delete all temporary files that were created to support processing."""
        for filename in os.listdir(os.getcwd()):
            if filename[:9] == 'TEMP_DEL_':
                os.remove(filename)


class Game(object):
    """A collection of data about a game whose text is in the Wikipedia corpus."""

    def __init__(self, pageid, title, year, intro_text_html, raw_text, categories_str):
        """Initialize a Game object."""
        self.pageid = pageid  # We use the unique Wikipedia pageid as a unique identifier
        self.wikipedia_url = 'https://en.wikipedia.org/?curid={pageid}'.format(pageid=pageid)
        # Remove artifacts like ' (video game)' from title
        self.title = title[:title.index('(') - 1] if '(' in title else title
        self.year = int(year) if year else 'upcoming'
        self.wikipedia_summary = intro_text_html
        self.raw_text = raw_text
        self.categories = categories_str.split('|')
        self.platforms = None  # Gets set by Pipeline.attribute_all_platforms()
        self.primary_platform = None  # Gets set by Pipeline.attribute_primary_platforms()
        self.text = None  # Gets set by Pipeline.preprocess_corpus()
        self.alternate_titles = []  # Gets modified by Pipeline.cull_duplicates()
        self.alternate_years = []  # Gets modified by Pipeline.cull_duplicates()
        self.lsa_vector = None  # Gets set by Pipeline.derive_lsa_model()
        # Fifty most/least-related games to this game; gets set by Pipeline.attribute_most_and_least_related_games()
        self.related_games = None
        self.unrelated_games = None
        self.three_dimensional_coordinates = None  # Gets set by Pipeline.derive_3d_coordinates()

    def __str__(self):
        """Return string representation."""
        return "{title} ({year})".format(title=self.title, year=self.year)


if __name__ == '__main__':
    THRESHOLD_FOR_DOCUMENT_LENGTH = 250  # 250 words
    K = 207  # Going with the k we empirically determined for the initial GameNet roll-out
    PATH_TO_CORPUS_DIR_WITH_TSV_FILES = '/Users/jamesryan/Desktop/gamespace/wiki_corpus_1_19_2017'
    PATH_TO_WRITE_GAMENET_FILES_TO = '/Users/jamesryan/Downloads'
    PATH_TO_WRITE_GAMESPACE_FILES_TO = '/Users/jamesryan/Downloads'
    pipeline = Pipeline()
