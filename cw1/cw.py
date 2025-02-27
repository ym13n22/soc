import numpy as np
import logging

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('Logging started')


def load_data(file):
    """ Load user-item rating matrix from CSV file """
    data = np.loadtxt(file, delimiter=',', dtype=np.float32)  # 修正溢出问题
    num_users = int(np.max(data[:, 0]))
    num_items = int(np.max(data[:, 1]))

    rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    for row in data:
        user, item, rating = int(row[0]) - 1, int(row[1]) - 1, row[2]
        rating_matrix[user, item] = rating

    return rating_matrix


def train_model(train_data):
    """ Train User-Based CF model by computing user-user similarity matrix using adjusted cosine similarity """
    num_users = train_data.shape[0]

    # 计算用户的平均评分，并使用 np.nan_to_num 处理 NaN 值
    user_mean = np.true_divide(train_data.sum(1), (train_data != 0).sum(1))
    user_mean = np.nan_to_num(user_mean)  # 修复 NaN 问题

    user_sim_matrix = np.zeros((num_users, num_users), dtype=np.float32)

    for u1 in range(num_users):
        for u2 in range(num_users):
            if u1 == u2:
                user_sim_matrix[u1, u2] = 1
                continue

            common_items = np.where((train_data[u1] > 0) & (train_data[u2] > 0))[0]
            if len(common_items) == 0:
                continue

            r_u1 = train_data[u1, common_items]
            r_u2 = train_data[u2, common_items]
            r_mean_u1 = user_mean[u1]
            r_mean_u2 = user_mean[u2]

            numerator = np.sum((r_u1 - r_mean_u1) * (r_u2 - r_mean_u2))
            denominator = np.sqrt(np.sum((r_u1 - r_mean_u1) ** 2)) * np.sqrt(np.sum((r_u2 - r_mean_u2) ** 2))

            if denominator > 0:
                user_sim_matrix[u1, u2] = numerator / denominator
            else:
                user_sim_matrix[u1, u2] = 0  # 避免 NaN 值

    return user_sim_matrix, user_mean


def predict_ratings(test_data, user_sim_matrix, user_mean):
    """ Predict ratings using User-Based CF with adjusted cosine similarity """
    num_users, num_items = test_data.shape
    predictions = np.zeros((num_users, num_items), dtype=np.float32)

    for user in range(num_users):
        for item in range(num_items):
            if test_data[user, item] == 0:  # 只预测未评分项
                sim_scores = user_sim_matrix[user]
                similar_users = np.where(test_data[:, item] > 0)[0]

                # 确保相似用户的平均评分不是 NaN
                valid_users = similar_users[~np.isnan(user_mean[similar_users])]

                if len(valid_users) > 0:
                    numerator = np.sum(
                        sim_scores[valid_users] * (test_data[valid_users, item] - user_mean[valid_users]))
                    denominator = np.sum(np.abs(sim_scores[valid_users]))

                    if denominator > 0:
                        predictions[user, item] = user_mean[user] + (numerator / denominator)
                    else:
                        predictions[user, item] = user_mean[user]  # 不能计算时，使用用户均值
                else:
                    predictions[user, item] = user_mean[user]  # 没有相似用户时，使用用户均值

    return predictions


def serialize_predictions(output_file, prediction_matrix):
    """ Save predictions to CSV file """
    np.savetxt(output_file, prediction_matrix, delimiter=',', fmt='%.2f')


if __name__ == '__main__':
    train = load_data('train_100k_withratings.csv')
    test = load_data('test_100k_withoutratings.csv')

    user_sim_matrix, user_mean = train_model(train)
    predictions = predict_ratings(test, user_sim_matrix, user_mean)

    serialize_predictions('submission.csv', predictions)
    logger.info('Predictions saved to submission.csv')
