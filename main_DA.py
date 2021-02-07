# LSTM
import tensorflow as tf
import numpy as np
import sys
import os
from LSTM_DA import LSTM_DA
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def get_batches(x, y, batch_size):
    #batch_num = int(x.shape[0] / batch_size)
    #c = 0
    #input_batches = []
    #target_batches = []
    #for i in range(batch_num - 1):
    #    input_batches.append(x[c: c + batch_size])
     #   target_batches.append(y[c: c + batch_size])
     #   c += batch_size
    #input_batches.append(x[c:])
    #target_batches.append(y[c:])
    indices = np.random.randint(0, x.shape[0], batch_size)
    input_batch = x[indices]
    target_batch = y[indices]
    return input_batch, target_batch#input_batches, target_batches


def normalize_feat(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = x[i][j] / np.linalg.norm(x[i][j])
    return x


def get_metrics(target, pred):
    # Assume target = [N x seq_len]
    #MAE = np.mean(np.abs(target - pred))
    #RMSE = np.sqrt(np.mean(np.square(target - pred)))
    label_acc = np.mean(np.equal(np.argmax(target, 1), np.argmax(pred, 1)))
    #label_precision = precision_score(np.argmax(target, 1), np.argmax(pred, 1))
    #label_recall = recall_score(np.argmax(target, 1), np.argmax(pred, 1))
    auc = roc_auc_score(np.argmax(target, 1), np.amax(pred,1))
    cm = confusion_matrix(np.argmax(target, 1), np.argmax(pred, 1))
    class_report = classification_report(np.argmax(target, 1), np.argmax(pred, 1))
    #label_acc = np.mean(float(correct_label_pred))
    return label_acc, auc,cm, class_report#MAE, RMSE


def training(training_input_s, training_input_t, training_target_s, training_target_t, learning_rate, training_epochs, hidden_dim, train, training_mode, batch_size, num_steps, model_path):
    print("Train data is loaded!")

    input_dim = training_input_s.shape[2]
    output_dim = training_target_s.shape[1]
    
    if training_mode == "dann":
        dann = 1
    else:
        dann = 0

    lstm = LSTM_DA(input_dim, output_dim, hidden_dim, train, dann)

    global_step = tf.contrib.framework.get_or_create_global_step()

    pred_loss = lstm.get_loss_gy()
    domain_loss = lstm.get_loss_gd()
    total_loss = tf.add(pred_loss, domain_loss)

    domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                               np.tile([0., 1.], [batch_size // 2, 1])])

    correct_label_pred = tf.equal(tf.argmax(lstm.classify_labels, 1), tf.argmax(lstm.label_pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(lstm.target_domain, 1), tf.argmax(lstm.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    # Setting up the optimizer
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    regular_train_op = tf.train.AdamOptimizer(learning_rate).minimize(pred_loss)
    dann_train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            p_acc = np.zeros(num_steps)
            d_acc = np.zeros(num_steps)
            for i in range(num_steps):
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                
                if dann == 1:
                    X0, y0 = get_batches(training_input_s, training_target_s, batch_size // 2)
                    X1, y1 = get_batches(training_input_t, training_target_t, batch_size // 2)
                    X = np.vstack([X0, X1])
                    y = np.vstack([y0, y1])
                    
                    _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                        [dann_train_op, total_loss, domain_loss, pred_loss, \
                         domain_acc, label_acc],
                        feed_dict={lstm.input: X, lstm.target: y, \
                            lstm.target_domain: domain_labels, lstm.l_coef: l})
                else:
                    if training_mode == "source":
                        X, y = get_batches(training_input_s, training_target_s, batch_size)
                    elif training_mode == "target":
                        X, y = get_batches(training_input_t, training_target_t, batch_size)
                    _, batch_loss, p_acc = sess.run([regular_train_op, pred_loss, label_acc],
                                         feed_dict={lstm.input: X, lstm.target: y, 
                                                    lstm.l_coef: l})
            if dann == 1:
                print('Epoch: %d, Mean_Pred_Accuracy: %f, Mean_Domain_Accuracy: %f' %
                  (epoch, np.mean(p_acc), np.mean(d_acc)))
            else:
                print('Epoch: %d, Mean_Pred_Accuracy: %f' %
                  (epoch, np.mean(p_acc)))

            if epoch % training_epochs == 0:
                saver.save(sess, os.path.join(model_path, 'checkpoint'), global_step=global_step)

        print("Training is over!")


def testing(test_input, test_target, hidden_dim, key, model_path):
    input_dim = test_input.shape[2]
    output_dim = test_target.shape[1]

    lstm_load = LSTM_DA(input_dim, output_dim, hidden_dim, key, 0)
    model_file = tf.train.latest_checkpoint(model_path)
    if model_file is None:
        print('No model found')
        sys.exit()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_file)
        pred = sess.run(lstm_load.get_outputs_gy(), \
                                feed_dict={lstm_load.input: test_input, \
                                lstm_load.target: test_target})
        target_acc, auc, cm, class_report = get_metrics(test_target, pred)
        print('Test Accuracy: %f' % (target_acc))
        #print('Test Precision: %f' % (target_precision))
        #print('Test Recall: %f' % (target_recall))
        print('Test AUC: %f' % (auc))
        print(cm)
        print(class_report)
        #print('Test RMSE: %f' % (rmse))
        np.save('preds.npy', pred)
        np.save('test_target.npy', test_target)


def main(argv):
    # Choose 1 for training and 0 for test
    # You can take the values for input parameters using sys.argv 
    train = 0  # int(sys.argv[1])
    training_mode = "source"
    data_source = np.load('data_source.npy')
    data_target = np.load('data_target.npy')
    data_source = normalize_feat(data_source)
    data_target = normalize_feat(data_target)
    target_source = np.load('target_source.npy')
    target_target = np.load('target_target.npy')
    #np.random.seed(123)
    random_indices_s = np.random.RandomState(seed=123).permutation(len(data_source))
    random_indices_t = np.random.RandomState(seed=123).permutation(len(data_target))
    data_source = data_source[random_indices_s]
    data_target = data_target[random_indices_t]
    target_source = target_source[random_indices_s]
    target_target = target_target[random_indices_t]
    training_size_s = int(0.7 * len(data_source))
    training_size_t = int(0.7 * len(data_target))
    if train:
        training_input_s, training_target_s = data_source[:training_size_s], target_source[:training_size_s]
        training_input_t, training_target_t = data_target[:training_size_t], target_target[:training_size_t]
        batch_size = 128
        num_steps = 100
        learning_rate = 3e-4  # float(sys.argv[3])
        training_epochs = 10  # int(sys.argv[4])
        hidden_dim = 24 # int(sys.argv[5])
        model_path = 'C:/Users/Sezin/Desktop/master/LSTM_ModelTrial/models/dann_lstm'  # str(sys.argv[7])
        training(training_input_s, training_input_t, training_target_s, training_target_t, learning_rate, training_epochs, hidden_dim, train, training_mode, batch_size, num_steps, model_path)
    else:
        test_input_t, test_target_t = data_target[training_size_t:], target_target[training_size_t:]
        hidden_dim = 24  # int(sys.argv[3])
        model_path = 'C:/Users/Sezin/Desktop/master/LSTM_ModelTrial/models/dann_lstm'  # str(sys.argv[4])
        testing(test_input_t, test_target_t, hidden_dim, train, model_path)


if __name__ == "__main__":
    main(sys.argv[1:])

