import numpy as np
import logging

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('Logging started')


def load_data(file, has_ratings=True):
    """ Load data from CSV file """
    data = np.loadtxt(file, delimiter=',', dtype=np.float32)  # 处理溢出问题
    if has_ratings:
        return data  # 训练集包含评分
    else:
        return data  # 测试集没有评分列，保持原样


def create_rating_matrix(train_data):
    """ Create a user-item rating matrix """
    num_users = int(np.max(train_data[:, 0]))
    num_items = int(np.max(train_data[:, 1]))
    rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    for row in train_data:
        user, item, rating = int(row[0]) - 1, int(row[1]) - 1, row[2]
        rating_matrix[user, item] = rating
    return rating_matrix


def train_model(train_matrix):
    """ Train User-Based CF model by computing user-user similarity matrix using adjusted cosine similarity """
    num_users = train_matrix.shape[0]
    user_mean = np.true_divide(train_matrix.sum(1), (train_matrix != 0).sum(1))
    user_mean = np.nan_to_num(user_mean)  # 处理 NaN
    user_sim_matrix = np.zeros((num_users, num_users), dtype=np.float32)

    for u1 in range(num_users):
        for u2 in range(num_users):
            if u1 == u2:
                user_sim_matrix[u1, u2] = 1
                continue
            common_items = np.where((train_matrix[u1] > 0) & (train_matrix[u2] > 0))[0]
            if len(common_items) == 0:
                continue
            r_u1, r_u2 = train_matrix[u1, common_items], train_matrix[u2, common_items]
            numerator = np.sum((r_u1 - user_mean[u1]) * (r_u2 - user_mean[u2]))
            denominator = np.sqrt(np.sum((r_u1 - user_mean[u1]) ** 2)) * np.sqrt(np.sum((r_u2 - user_mean[u2]) ** 2))
            user_sim_matrix[u1, u2] = numerator / denominator if denominator > 0 else 0

    return user_sim_matrix, user_mean


def predict_ratings(test_data, train_matrix, user_sim_matrix, user_mean):
    """ Predict ratings using User-Based CF """
    predictions = []
    for row in test_data:
        user, item, timestamp = int(row[0]), int(row[1]), int(row[2])
        user_idx, item_idx = user - 1, item - 1
        sim_scores = user_sim_matrix[user_idx]
        similar_users = np.where(train_matrix[:, item_idx] > 0)[0]
        valid_users = similar_users[~np.isnan(user_mean[similar_users])]
        if len(valid_users) > 0:
            numerator = np.sum(sim_scores[valid_users] * (train_matrix[valid_users, item_idx] - user_mean[valid_users]))
            denominator = np.sum(np.abs(sim_scores[valid_users]))
            rating_pred = user_mean[user_idx] + (numerator / denominator) if denominator > 0 else user_mean[user_idx]
        else:
            rating_pred = user_mean[user_idx]
        predictions.append([user, item, rating_pred, timestamp])
    return predictions


def serialize_predictions(output_file, predictions):
    """ Save predictions to CSV file """
    np.savetxt(output_file, predictions, delimiter=',', fmt=['%d', '%d', '%.2f', '%d'])
    logger.info(f'Predictions saved to {output_file}')


if __name__ == '__main__':
    train_data = load_data('train_100k_withratings.csv')
    test_data = load_data('test_100k_withoutratings.csv', has_ratings=False)
    train_matrix = create_rating_matrix(train_data)
    user_sim_matrix, user_mean = train_model(train_matrix)
    predictions = predict_ratings(test_data, train_matrix, user_sim_matrix, user_mean)
    serialize_predictions('submission.csv', predictions)
