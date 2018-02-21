import argparse
import io
import json
import numpy as np
import os
import random
import redis
import sys
import tensorflow as tf
import unicodecsv

from collections import defaultdict
from jack import readers
from jack.io import embeddings
from random import randint
from tqdm import tqdm

from ir.search_engine import SearchEngine
from playground.datareader import format_paths
from qa.nouns import pre_extract_nouns, SpacyNounParser, NltkNounParser
from qa.question import Question
from rc.utils import get_rc_answers, get_cached_rc_answers
from shared.utils import trim_index, form_query, get_document_for_query


def discount_rewards(r, gamma):
    """Discount history of rewards by gamma."""
    discounted_r = np.zeros(len(r))
    running_add = 0
    # Most recent reward is last
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def check_answer(answer, question, incorrect_answers_this_episode, e, corrects, incorrects):
    reward = -0.01
    if answer.text.lower() in question.candidates_lower:
        if answer.text.lower() == question.answer.lower():
            print(e, ': Found correct answer', answer.text)
            corrects.append(e)
            reward = 1
        # If the current answer is not correct and we have not submitted it before
        elif answer.text.lower() not in incorrect_answers_this_episode:
            reward = 0
            print(e, ': Found incorrect answer candidate', answer.text)
            incorrect_answers_this_episode.append(answer.text.lower())
            if e not in incorrects:
                incorrects.append(e)
    return reward, incorrect_answers_this_episode, corrects, incorrects


def vocab_for_dataset(dataset):
    vocab = set()
    counts = defaultdict(int)
    for question in dataset:
        for doc in question['supports']:
            for w in doc.split():
                vocab.add(w)
                counts[w] += 1
    return vocab, counts


def idf_for_dataset(dataset, use_lowercase=True):
    # TODO: use log(N/n_t)
    df = defaultdict(int)
    for question in dataset:
        for doc in question['supports'] + [question['query']]:
            for word in doc.split():
                if use_lowercase:
                    word = word.lower()
                df[word] += 1
    for word in df.keys():
        df[word] = 1.0/df[word]
    return defaultdict(lambda: 1, df)


def write_summary(summary_writer, running_reward, episode, episode_length):
    ep_summary = tf.Summary()
    ep_summary.value.add(simple_value=running_reward, tag='reward')
    ep_summary.value.add(simple_value=episode_length, tag='episode_length')
    summary_writer.add_summary(ep_summary, episode)
    summary_writer.flush()


def unit_sphere(var_matrix, norm=1.0, axis=1):
    # Norm for each word embedding vector
    row_norms = np.sqrt(np.sum(np.square(var_matrix), axis=axis))
    # Divide vectors by their norms to obtain unit vectors
    scaled = var_matrix * np.expand_dims(norm / row_norms, axis=axis)
    return scaled


class GloveLookup:
    def __init__(self, path, dim, dataset, idf_from_file='rl/idf_weights_lower.json'):
        self.emb_dim = dim
        print('Loading GloVe...')
        vocab, _ = vocab_for_dataset(dataset)
        print('Train vocab length', len(vocab))
        if idf_from_file is not None and os.path.exists(idf_from_file):
            print('Loaded IDF weights from', idf_from_file)
            self.idf = json.load(io.open(idf_from_file, 'r', encoding='utf-8'))
        else:
            print('Computing IDF for %i question items...' % len(dataset))
            self.idf = idf_for_dataset(dataset)
            with io.open(idf_from_file, 'w', encoding='utf8') as f:
                json.dump(self.idf, f, ensure_ascii=False)

        self.word2idx, self.lookup = embeddings.glove.load_glove(open(path, 'rb'))
        num_OOV_words = 0
        for word in tqdm(vocab):
            if word.lower() not in self.word2idx:
                idx = len(self.word2idx)
                num_OOV_words += 1
                self.word2idx[word.lower()] = idx
                if idx > len(self.lookup) - 1:
                    self.lookup.resize([self.lookup.shape[0] + 50000, self.lookup.shape[1]])
                self.lookup[idx] = np.random.normal(0.0, 1.0, size=[1, dim])
        # TODO: learn linear transformation for task
        print('Initialised', num_OOV_words, 'new words')
        self.OOV = np.zeros(dim)
        self.OOV[0] = 1
        print(self.lookup[:len(self.word2idx), :].shape)
        self.lookup = unit_sphere(self.lookup[:len(self.word2idx), :])
        self.state_str_history = []
        self.state_history = []
        self.avg_state_history = []
        self.dataset_len = len(dataset)

    def save_history_to_csv(self):
        print('History length', len(self.state_history), len(self.state_str_history),
              len(self.avg_state_history))

        with open('rl/tf_idf_states-%i.csv' % self.dataset_len, 'wb') as f:
            writer = unicodecsv.writer(f)
            for s in self.state_history:
                writer.writerow(list(s))
        with open('rl/avg_states-%i.csv' % self.dataset_len, 'wb') as f:
            writer = unicodecsv.writer(f)
            for s in self.avg_state_history:
                writer.writerow(list(s))
        with open('rl/state-strs-%i.txt' % self.dataset_len, 'w') as f:
            for s in self.state_str_history:
                f.write(str(s) + '\n')

    def lookup_word(self, word):
        if word.lower() in self.word2idx:
            return self.lookup[self.word2idx[word.lower()]]
        return self.OOV

    def lookup_doc_tf_idf(self, doc, tf):
        words = doc.split()
        for w in words:
            if w.lower() not in tf:
                print(w.lower(), 'not in tf')
            if w.lower() not in self.idf:
                print(doc)
                print(w.lower(), 'not in idf')
        return np.mean([tf[w.lower()] * self.idf[w.lower()] * self.lookup_word(w) for w in words],
                       axis=0)

    def lookup_doc_avg(self, doc):
        words = doc.split()
        return np.mean([self.lookup_word(w) for w in words], axis=0)

    def embed_state(self, s):
        tfs = []
        for elem in s:
            tfs.append(defaultdict(int))
            for word in elem.split():
                tfs[-1][word.lower()] += 1
        tf_idf_state = np.stack([self.lookup_doc_tf_idf(s[e], tfs[e]) for e in range(len(s))],
                                axis=0).flatten()
        naive_state = np.stack([self.lookup_doc_avg(elem) for elem in s], axis=0).flatten()
        self.state_history.append(list(tf_idf_state))
        self.avg_state_history.append(list(naive_state))
        self.state_str_history.append(s)
        return tf_idf_state


def eval_random_baseline(dataset, search_engine, nouns, reader, redis_server, actions, max_queries,
                         confidence_threshold, verbose, num_episodes, run_id):
    print(len(dataset), 'questions')
    total_reward = []
    total_length = []
    corrects = []
    incorrects = []
    verbose_list = []

    with tf.Session() as sess:
        summary_writer = None
        if run_id is not None:
            summary_writer = tf.summary.FileWriter('rl/summaries/train-' + run_id, sess.graph)

        for e in range(num_episodes):
            question = Question(dataset[e % len(dataset)])
            if verbose or e in verbose_list:
                print('\n' + str(e), ':', question.query, '(', question.answer, ')')
            _, subj0 = question.query.split()[0], ' '.join(question.query.split()[1:]).lower()
            query0 = actions[0] + ' ' + subj0 + '?'
            top_idx = search_engine.rank_docs(question.id, query0, topk=len(question.supports))[-1]
            d0 = question.supports[top_idx]

            # Send query to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, query0, d0)
            else:
                rc_answers, _ = get_cached_rc_answers(reader, query0, d0, redis_server)
            ans0 = rc_answers[0]
            running_reward = 0
            ep_history = []

            subj_t = subj0
            incorrect_answers_this_episode = []
            # Store past queries asked -> documents retrieved mappings
            queries_asked = defaultdict(list)
            for t in range(max_queries):
                top_idx = None
                while top_idx is None:
                    # Randomly pick an action
                    a_t = np.random.randint(len(actions))
                    top_idx, subj_t, query_t = get_document_for_query(
                        actions[a_t], subj_t, search_engine, question, nouns, queries_asked)
                queries_asked[query_t].append(top_idx)
                ep_history.append(a_t)
                d_t = question.supports[top_idx]

                if verbose or e in verbose_list:
                    print(query_t, '\n\t->', top_idx)
                # Send query to RC module
                if redis_server is None:
                    rc_answers = get_rc_answers(reader, query_t, d_t)
                else:
                    rc_answers, _ = get_cached_rc_answers(reader, query_t, d_t, redis_server)
                ans_t = rc_answers[0]

                r, incorrect_answers_this_episode, corrects, incorrects = check_answer(
                    ans_t, question, incorrect_answers_this_episode, e, corrects, incorrects)
                if ans_t.score > confidence_threshold:
                    subj_t = ans_t.text.lower()
                    if verbose or e in verbose_list:
                        print('\t->', subj_t, '(', ans_t.score, ')')
                else:
                    if verbose or e in verbose_list:
                        print('\t->', subj_t, '(', ans_t.score, ') -> pick at random')
                    # Pick a noun phrase at random from top document
                    rand_draw = randint(0, len(nouns[question.id][top_idx])-1)
                    subj_t = nouns[question.id][top_idx][rand_draw].lower()
                running_reward += r
                ep_length = t
                if r == 1:
                    print('\tAction history:', ep_history)
                    break

            if summary_writer is not None:
                write_summary(summary_writer, running_reward, e, ep_length)
            total_reward.append(running_reward)
            total_length.append(ep_length)

            '''if e % 10 == 0:
                print('  Reward history', np.mean(total_reward[-100:]))
                print('  Correct answers', correct_answers, float(correct_answers)/(e+1))
                print('  Incorrect answers', incorrect_answers, float(incorrect_answers)/(e+1))'''

        print('Correct answers final', len(corrects), float(len(corrects))/num_episodes)
        print('Incorrect answers final', len(incorrects), float(len(incorrects))/num_episodes)


class Reinforce:
    """Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb."""
    def __init__(self, lr, state_shape, action_shape, hidden_sizes):
        # Policy network
        self.state_in = tf.placeholder(shape=[None, state_shape], dtype=tf.float32)
        hidden = self.state_in
        for h_size in hidden_sizes:
            hidden = tf.contrib.slim.fully_connected(hidden, h_size,
                                                     biases_initializer=None,
                                                     activation_fn=tf.nn.relu)
        self.output = tf.contrib.slim.fully_connected(hidden, action_shape,
                                                      activation_fn=tf.nn.softmax,
                                                      biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Training
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = (tf.range(0, tf.shape(self.output)[0])
                        * tf.shape(self.output)[1] + self.action_holder)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


def initialise(included_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--nltk', nargs='?', const=True, default=False, type=bool,
                        help='If True, use NLTK to parse nouns. If False, use Spacy.')
    parser.add_argument('--verbose', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out all mentions of the query subject.')
    parser.add_argument('--subset_size', default=None, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--num_items_to_eval', default=None, type=int,
                        help='If set and <= subset size, the number of instances to evaluate.')
    parser.add_argument('--k_most_common_only', type=int, default=None,
                        help='If set, only include the k most commonly occurring relation types.')
    parser.add_argument('--wikihop_version', type=str, default='1.1',
                        help='WikiHop version to use: one of {0, 1.1}.')
    parser.add_argument('--dev', nargs='?', const=True, default=False,
                        help='If True, build an index on dev data instead of train.')
    parser.add_argument('--parallel', nargs='?', const=True, default=False, type=bool,
                        help='If True, use NLTK to parse nouns. If False, use Spacy.')
    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.add_argument('--nocache', dest='cache', action='store_false')
    parser.add_argument('--conf_threshold', default=0.10, type=float,
                        help='Confidence threshold required to use ')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning Rate')
    parser.add_argument('--log_dir', type=str, default='./rl/summaries',
                        help='Directory for TensorBoard summaries.')
    parser.add_argument('--save_embs', const=True, default=False, type=bool,
                        help='If True, save state embedding vectors to CSV.')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Identifier for TensorBoard summary files.')
    parser.add_argument('--random_agent', action='store_true', default=False,
                        help='If True, randomly pick out of available actions.')
    parser.add_argument('--trim', dest='trim_index', action='store_true')
    parser.add_argument('--notrim', dest='trim_index', action='store_false')
    parser.set_defaults(cache=True, trim_index=True)
    args = parser.parse_args()

    subset_id, data_path, index_filename, nouns_path = format_paths(args)
    redis_server = None
    if args.cache:
        redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)

    print('Initialising...')
    with open(data_path) as dataset_file:
        dataset = json.load(dataset_file)
    if args.subset_size is not None:
        dataset = dataset[:args.subset_size]
    search_engine = SearchEngine(dataset, load_from_path=index_filename)

    if args.nltk:
        print('Extracting NTLK nouns...')
        noun_parser_class = NltkNounParser
    else:
        print('Extracting Spacy nouns...')
        noun_parser_class = SpacyNounParser
    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    nouns = pre_extract_nouns(dataset, nouns_path, noun_parser_class=noun_parser_class)
    reader = readers.reader_from_file('./rc/fastqa_reader')

    # Number of instances to test (if > subset size, repeat items)
    num_items_to_eval = len(dataset)
    if args.num_items_to_eval is not None:
        num_items_to_eval = min(len(dataset), args.num_items_to_eval)
    print('Evaluating', num_items_to_eval, 'questions')

    one_type_only = True  # Set to True to evaluate templates on a single relation type
    one_type_dataset = []
    if one_type_only:
        for q in dataset:
            if q['query'].split()[0] == included_type:
                one_type_dataset.append(q)

        dataset = one_type_dataset

    dataset = dataset[:num_items_to_eval]
    if args.trim_index:
        nouns, search_engine = trim_index(dataset, nouns, search_engine)

    return dataset, search_engine, nouns, reader, redis_server, args


def train(query_type, dataset, search_engine, nouns, reader, redis_server, args):
    if query_type == 'located_in_the_administrative_territorial_entity':
        actions = [['where is'],
                   ['municipality of'],
                   ['located'],
                   ['what kind of socks does', 'wear']]
    elif query_type == 'occupation':
        actions = ['employer of',
                   'colleague of',
                   'known for',
                   'work as']
    gamma = 0.95
    lr = args.lr
    h_sizes = [64, 32]
    # Make experiments repeatable
    random.seed(0)

    # Threshold above which to trust the reading comprehension module's answers
    confidence_threshold = args.conf_threshold
    max_queries = 25
    if args.num_items_to_eval is None:
        num_episodes = len(dataset)
    else:
        # Repeat if num_episodes > len(dataset)
        num_episodes = args.num_items_to_eval

    if args.random_agent:
        eval_random_baseline(dataset, search_engine, nouns, reader, redis_server, actions,
                             max_queries, confidence_threshold, args.verbose, num_episodes,
                             args.run_id)
        sys.exit()

    emb_dim = 50
    embs = GloveLookup('./data/GloVe/glove.6B.%id.txt' % emb_dim, emb_dim, dataset)

    if args.run_id is not None:
        run_id = args.run_id + '-lr' + str(lr)
        summaries_path = 'rl/summaries/%s' % run_id
        checkpoints_path = 'rl/checkpoints/%s/model-' % run_id
        for path in [summaries_path, checkpoints_path]:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

    tf.reset_default_graph()

    # State: subj0,  a_t-1,     subj_t-1, d_t-1,   ans_t-1 + history
    #        emb_dim, emb_dim, emb_dim,  emb_dim, emb_dim + TODO
    agent = Reinforce(lr=lr, state_shape=5*emb_dim, action_shape=len(actions), hidden_sizes=h_sizes)

    update_frequency = 50
    corrects = []
    incorrects = []
    incorrect_answers_this_episode = []

    print(len(dataset), 'questions')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.run_id is not None:
            summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
            saver = tf.train.Saver()
        reward_history = []
        ep_length_history = []

        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        for e in range(num_episodes):
            question = Question(dataset[e % len(dataset)])
            _, subj0 = question.query.split()[0], ' '.join(question.query.split()[1:]).lower()
            # First action taken with partial information: no a_t-1, subj_t-1, d_t-1, ans_t-1
            query0 = form_query(actions[0], subj0)
            # First query won't have been asked from top doc
            top_idx = search_engine.rank_docs(question.id, query0, topk=len(question.supports))[-1]
            d0 = question.supports[top_idx]

            # Send query to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, query0, d0)
            else:
                rc_answers, _ = get_cached_rc_answers(reader, query0, d0, redis_server)
            ans0 = rc_answers[0]
            # Initial state
            s_prev = [subj0, ' '.join(actions[0]), subj0, d0, ans0.text]
            s_prev = np.hstack(embs.embed_state(s_prev))
            # Should first answer be checked?
            '''
            (r, correct_answers, incorrect_answers, incorrect_answers_this_episode,
             seen_incorrect_answer_this_episode) = (
                 check_answer(ans0, question, correct_answers, incorrect_answers,
                              incorrect_answers_this_episode, seen_incorrect_answer_this_episode,
                              e))
            '''
            ep_reward = 0
            subj_t = subj0

            # Store past queries asked -> documents retrieved mappings
            queries_asked = defaultdict(list)
            for t in range(max_queries):
                top_idx = None
                while top_idx is None:
                    # Pick action according to policy
                    a_distr = sess.run(agent.output, feed_dict={agent.state_in: [s_prev]})
                    a_t = np.random.choice(range(len(actions)), p=a_distr[0])
                    top_idx, subj_t, query_t = get_document_for_query(
                        actions[a_t], subj_t, search_engine, question, nouns, queries_asked)
                queries_asked[query_t].append(top_idx)
                d_t = question.supports[top_idx]

                # Send query to RC module
                if redis_server is None:
                    rc_answers = get_rc_answers(reader, query_t, d_t)
                else:
                    rc_answers, _ = get_cached_rc_answers(reader, query_t, d_t, redis_server)
                ans_t = rc_answers[0]

                (r, incorrect_answers_this_episode, corrects, incorrects) = (
                    check_answer(ans_t, question, incorrect_answers_this_episode, e, corrects,
                                 incorrects))
                # TODO: alternative: form_query(actions[a_t], subj_t)
                s_t = [subj0, ' '.join(actions[a_t]), subj_t, d_t, ans_t.text]
                s_t = np.hstack(embs.embed_state(s_t))

                history_frame = np.expand_dims([s_prev, a_t, r, s_t], axis=0)
                if t == 0:
                    ep_history = history_frame
                else:
                    ep_history = np.append(ep_history, history_frame, axis=0)
                s_prev = s_t
                if ans_t.score > confidence_threshold:
                    subj_t = ans_t.text.lower()
                else:
                    # Pick a noun phrase at random from top document
                    rand_draw = randint(0, len(nouns[question.id][top_idx])-1)
                    subj_t = nouns[question.id][top_idx][rand_draw].lower()
                ep_reward += r
                ep_length = t
                if r == 1:
                    print('\tAction history:', ep_history[:, 1])
                    break

            ep_history[:, 2] = discount_rewards(ep_history[:, 2], gamma)
            feed_dict = {agent.reward_holder: ep_history[:, 2],
                         agent.action_holder: ep_history[:, 1],
                         agent.state_in: np.vstack(ep_history[:, 0])}
            grads = sess.run(agent.gradients, feed_dict=feed_dict)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if e % update_frequency == 0 and e != 0:
                feed_dict = dict(zip(agent.gradient_holders, gradBuffer))
                _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

            if args.run_id is not None:
                write_summary(summary_writer, ep_reward, e, ep_length)
            reward_history.append(ep_reward)
            ep_length_history.append(len(ep_history))

            if e % 10 == 0:
                print('  Agent output',
                      sess.run(agent.output, feed_dict={agent.state_in: [s_prev]}))
                print('  Reward history', np.mean(reward_history[-100:]))
                print('  Correct answers', len(corrects), float(len(corrects))/(e+1))
                print('  Incorrect answers', len(incorrects), float(len(incorrects))/(e+1))
                if args.run_id is not None:
                    saver.save(sess, checkpoints_path + str(e) + '.cptk')
    if args.save_embs:
        embs.save_history_to_csv()


if __name__ == "__main__":
    query_type = 'located_in_the_administrative_territorial_entity'  # 'occupation'
    dataset, search_engine, nouns, reader, redis_server, args = initialise(query_type)
    train(query_type, dataset, search_engine, nouns, reader, redis_server, args)
