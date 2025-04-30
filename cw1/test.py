import numpy as np
import logging
from sklearn.model_selection import train_test_split

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


def create_rating_matrix_with_split(train_data, test_data):
    """ Create a user-item rating matrix, ensuring it covers all itemp IDs in test_data """
    num_users = int(np.max(train_data[:, 0]))  # 计算最大用户 ID
    num_items_train = int(np.max(train_data[:, 1]))  # 计算训练数据最大 item ID
    num_items_test = int(np.max(test_data[:, 1])) if test_data is not None else 0  # 计算测试数据最大 item ID
    num_items = max(num_items_train, num_items_test)  # 选择最大的 item ID 确保矩阵足够大

    rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    for row in train_data:
        user, item, rating = int(row[0]) - 1, int(row[1]) - 1, row[2]
        rating_matrix[user, item] = rating

    return rating_matrix

def create_rating_matrix(train_data):
    """ Create a user-item rating matrix """
    num_users = int(np.max(train_data[:, 0]))
    num_items = int(np.max(train_data[:, 1]))
    rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    for row in train_data:
        user, item, rating = int(row[0]) - 1, int(row[1]) - 1, row[2]
        rating_matrix[user, item] = rating
    return rating_matrix


def train_model(train_matrix, significance_threshold=50, case_amplification=1.1):
    """ 计算用户-用户相似度矩阵（调整余弦相似度）并优化 """
    num_users = train_matrix.shape[0]
    user_mean = np.true_divide(train_matrix.sum(1), (train_matrix != 0).sum(1))
    user_mean = np.nan_to_num(user_mean)
    user_sim_matrix = np.zeros((num_users, num_users), dtype=np.float32)

    for u1 in range(num_users):
        for u2 in range(num_users):
            if u1 == u2:
                user_sim_matrix[u1, u2] = 1
                continue
            common_items = np.where((train_matrix[u1] > 0) & (train_matrix[u2] > 0))[0]
            num_common = len(common_items)
            if num_common == 0:
                continue

            r_u1, r_u2 = train_matrix[u1, common_items], train_matrix[u2, common_items]
            numerator = np.sum((r_u1 - user_mean[u1]) * (r_u2 - user_mean[u2]))
            denominator = np.sqrt(np.sum((r_u1 - user_mean[u1]) ** 2)) * np.sqrt(np.sum((r_u2 - user_mean[u2]) ** 2))
            sim = (numerator / denominator) if denominator > 0 else 0

            # 1. 显著性加权 (Significance Weighting)
            weight_factor = num_common / (num_common + significance_threshold)  # 平滑处理
            sim *= weight_factor

            sim = np.sign(sim) * (abs(sim) ** case_amplification)

            # 2. 案例放大 (Case Amplification)
            user_sim_matrix[u1, u2] = sim
            user_sim_matrix[u2, u1] = sim


    return user_sim_matrix, user_mean


def predict_ratings_with_split(test_data, train_matrix, user_sim_matrix, user_mean, similarity_threshold=0.005,
                               neighborhood_size=30):
    """ 使用邻域选择优化的 User-Based CF 进行评分预测，并考虑高方差物品赋予更高权重 """
    predictions = []
    for row in test_data:
        user, item, timestamp = int(row[0]), int(row[1]), int(row[3])
        user_idx, item_idx = user - 1, item - 1
        sim_scores = user_sim_matrix[user_idx]
        similar_users = np.where(train_matrix[:, item_idx] > 0)[0]

        # 3. 邻域选择 (Neighborhood Selection)
        valid_users = similar_users[sim_scores[similar_users] > similarity_threshold]
        if len(valid_users) > neighborhood_size:
            valid_users = valid_users[np.argsort(sim_scores[valid_users])[-neighborhood_size:]]

        if len(valid_users) == 0:
            rating_pred = user_mean[user_idx]
        else:
            # **计算物品评分方差并加权**
            item_variance = np.var(train_matrix[train_matrix[:, item_idx] > 0, item_idx])  # 仅考虑已评分项
            weight = 1 + np.log1p(item_variance)

            # **修改评分计算公式**
            numerator = np.sum(sim_scores[valid_users] * (train_matrix[valid_users, item_idx] - user_mean[valid_users]) * weight)
            denominator = np.sum(np.abs(sim_scores[valid_users]) * weight)

            rating_pred = user_mean[user_idx] + (numerator / denominator) if denominator > 0 else user_mean[user_idx]

        predictions.append([user, item, rating_pred, timestamp])
    return predictions



def predict_ratings(test_data, train_matrix, user_sim_matrix, user_mean, similarity_threshold=0.0001,
                    neighborhood_size=40):
    """ 使用用户相似度进行评分预测 """
    predictions = []

    for row in test_data:
        user, item, timestamp = int(row[0]), int(row[1]), int(row[2])
        user_idx, item_idx = user - 1, item - 1
        sim_scores = user_sim_matrix[user_idx]
        similar_users = np.where(train_matrix[:, item_idx] > 0)[0]

        # 3. 邻域选择 (Neighborhood Selection)
        valid_users = similar_users[sim_scores[similar_users] > similarity_threshold]
        if len(valid_users) > neighborhood_size:
            valid_users = valid_users[np.argsort(sim_scores[valid_users])[-neighborhood_size:]]

        if len(valid_users) == 0:
            rating_pred = user_mean[user_idx]
        else:
            # **计算物品评分方差并加权**
            item_variance = np.var(train_matrix[train_matrix[:, item_idx] > 0, item_idx])  # 仅考虑已评分项
            weight = 1 + np.log1p(item_variance)

            # **修改评分计算公式**
            numerator = np.sum(
                sim_scores[valid_users] * (train_matrix[valid_users, item_idx] - user_mean[valid_users]) * weight)
            denominator = np.sum(np.abs(sim_scores[valid_users]) * weight)

            rating_pred = user_mean[user_idx] + (numerator / denominator) if denominator > 0 else user_mean[user_idx]

        predictions.append([user, item, rating_pred, timestamp])
    return predictions


def evaluate_model(predictions, test_data):
    """ Compute MAE for evaluation """
    test_ratings = test_data[:, 2]
    predicted_ratings = np.array([pred[2] for pred in predictions])
    mae = np.mean(np.abs(test_ratings - predicted_ratings))
    logger.info(f'Model MAE: {mae:.4f}')
    return mae


def serialize_predictions(output_file, predictions):
    """ Save predictions to CSV file """
    np.savetxt(output_file, predictions, delimiter=',', fmt=['%d', '%d', '%.2f', '%d'])
    logger.info(f'Predictions saved to {output_file}')


if __name__ == '__main__':
    data = load_data('train_100k_withratings.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_matrix = create_rating_matrix_with_split(train_data, test_data)  # 传入 test_data 确保足够的列
    user_sim_matrix, user_mean = train_model(train_matrix)
    predictions = predict_ratings_with_split(test_data, train_matrix, user_sim_matrix, user_mean)
    evaluate_model(predictions, test_data)
    serialize_predictions('submission1.csv', predictions)
    train_data1 = load_data('train_100k_withratings.csv')
    test_data1 = load_data('test_100k_withoutratings.csv', has_ratings=False)
    train_matrix1 = create_rating_matrix(train_data1)
    user_sim_matrix1, user_mean1 = train_model(train_matrix1)
    predictions1 = predict_ratings(test_data1, train_matrix1, user_sim_matrix1, user_mean1)
    serialize_predictions('submission.csv', predictions1)