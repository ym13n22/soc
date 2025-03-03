import numpy as np
import logging

LOG_FORMAT = ('%(levelname)-s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('Logging started')


def load_data(file, has_ratings=True):
    """ Load data from CSV file """
    data = np.loadtxt(file, delimiter=',', dtype=np.float32)
    return data


def split_train_test(data, train_ratio=0.8):
    """ Split data into train and validation sets """
    np.random.shuffle(data)  # 随机打乱数据
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]


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
    """ Train Item-Based CF model by computing item-item similarity matrix using adjusted cosine similarity """
    num_items = train_matrix.shape[1]

    # 计算每个物品的评分均值
    num_ratings_per_item = (train_matrix != 0).sum(0)
    item_mean = np.divide(train_matrix.sum(0), num_ratings_per_item, where=num_ratings_per_item != 0)
    item_mean = np.nan_to_num(item_mean)  # 处理 NaN

    # 计算物品-物品相似度矩阵
    item_sim_matrix = np.zeros((num_items, num_items), dtype=np.float32)

    for i1 in range(num_items):
        for i2 in range(num_items):
            if i1 == i2:
                item_sim_matrix[i1, i2] = 1
                continue
            common_users = np.where((train_matrix[:, i1] > 0) & (train_matrix[:, i2] > 0))[0]
            if len(common_users) == 0:
                continue
            r_i1, r_i2 = train_matrix[common_users, i1], train_matrix[common_users, i2]
            numerator = np.sum((r_i1 - item_mean[i1]) * (r_i2 - item_mean[i2]))
            denominator = np.sqrt(np.sum((r_i1 - item_mean[i1]) ** 2)) * np.sqrt(np.sum((r_i2 - item_mean[i2]) ** 2))
            item_sim_matrix[i1, i2] = numerator / denominator if denominator > 0 else 0

    return item_sim_matrix, item_mean


def predict_ratings(test_data, train_matrix, item_sim_matrix, item_mean):
    """ Predict ratings using Item-Based CF """
    predictions = []
    for row in test_data:
        user, item = int(row[0]), int(row[1])
        user_idx, item_idx = user - 1, item - 1

        sim_scores = item_sim_matrix[item_idx]
        similar_items = np.where(train_matrix[user_idx, :] > 0)[0]
        valid_items = similar_items[~np.isnan(item_mean[similar_items])]

        if len(valid_items) > 0:
            numerator = np.sum(sim_scores[valid_items] * (train_matrix[user_idx, valid_items] - item_mean[valid_items]))
            denominator = np.sum(np.abs(sim_scores[valid_items]))
            rating_pred = item_mean[item_idx] + (numerator / denominator) if denominator > 0 else item_mean[item_idx]
        else:
            rating_pred = item_mean[item_idx]

        predictions.append([user, item, rating_pred])

    return predictions


def evaluate(predictions, true_data):
    """ Compute MAE (Mean Absolute Error) for evaluation """
    absolute_errors = []
    for pred, true in zip(predictions, true_data):
        user, item, rating_pred = pred
        true_rating = true[2]
        absolute_errors.append(abs(rating_pred - true_rating))

    mae = np.mean(absolute_errors)
    logger.info(f'Validation MAE: {mae:.4f}')
    return mae


def serialize_predictions(output_file, predictions):
    """ Save predictions to CSV file """
    np.savetxt(output_file, predictions, delimiter=',', fmt=['%d', '%d', '%.2f'])
    logger.info(f'Predictions saved to {output_file}')


if __name__ == '__main__':
    # 1. 加载训练数据
    full_train_data = load_data('train_100k_withratings.csv')

    # 2. 按 80/20 划分数据
    train_data, valid_data = split_train_test(full_train_data, train_ratio=0.8)

    # 3. 创建评分矩阵
    train_matrix = create_rating_matrix(train_data)

    # 4. 训练模型
    item_sim_matrix, item_mean = train_model(train_matrix)

    # 5. 在 validation set 上评估 MAE
    valid_predictions = predict_ratings(valid_data, train_matrix, item_sim_matrix, item_mean)
    evaluate(valid_predictions, valid_data)

    # 6. 预测测试数据（最终提交用）
    test_data = load_data('test_100k_withoutratings.csv', has_ratings=False)
    test_predictions = predict_ratings(test_data, train_matrix, item_sim_matrix, item_mean)

    # 7. 保存预测结果
    serialize_predictions('submission.csv', test_predictions)
