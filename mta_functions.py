import random
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import tee
from functools import reduce
from operator import mul

def remove_immediate_loops(path_string: str) -> str:
    """Remove immediate self-loops in a path."""
    path_list = path_string.split(' > ')
    if not path_list:
        return path_string
    new_path = [path_list[0]]
    for node in path_list[1:]:
        if node != new_path[-1]:
            new_path.append(node)
    return ' > '.join(new_path)

def expand_and_restructure_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the DataFrame so that each row represents a single user path and add conversion information.
    """
    expanded_rows = []
    for _, row in data.iterrows():
        path = row['path']
        total_conversions = row['total_conversions']
        total_nulls = row['total_null']
        conversion_value = row['total_conversion_value']
        
        # Add rows for converted paths
        for _ in range(total_conversions):
            expanded_rows.append({
                'path': path,
                'conversion': 1,
                'conversion_value': conversion_value / total_conversions if total_conversions > 0 else 0
            })
        
        # Add rows for non-converted paths
        for _ in range(total_nulls):
            expanded_rows.append({
                'path': path,
                'conversion': 0,
                'conversion_value': 0
            })
    
    return pd.DataFrame(expanded_rows)

def add_exposure_times(data: pd.DataFrame, path_column: str, exposure_column: str, exposure_every_second: bool = False) -> pd.DataFrame:
    """
    Add exposure times to the data.
    """
    def calculate_exposure_times(path_length):
        if exposure_every_second:
            return list(range(path_length))
        else:
            return sorted([random.randint(0, 30) for _ in range(path_length)], reverse=True)

    def process_path_string(path_string):
        path_list = path_string.split(' > ')
        return calculate_exposure_times(len(path_list))

    data[exposure_column] = data[path_column].apply(process_path_string)
    return data

def process_data(data):
    """Load and preprocess the data."""
    data['path'] = data['path'].apply(remove_immediate_loops)
    detail_data = expand_and_restructure_data(data)
    detail_data = add_exposure_times(detail_data, 'path', 'ts', exposure_every_second=False)
    
    # Filter out only conversion paths
    conv_data = detail_data[detail_data['conversion'] == 1]
    conv_multi_channel_data = conv_data[conv_data['path'].str.contains(' > ')]
    
    # Filter out only multi-channel paths, original data structure
    multi_channel_data = detail_data[detail_data['path'].str.contains(' > ')][['path', 'conversion', 'conversion_value']]
    return multi_channel_data, conv_multi_channel_data

def linear_attribution(conv_data: pd.DataFrame, path_column: str, conversion_value_column: str) -> dict:
    """
    Calculate linear attribution for each channel in the conversion paths.
    """
    unique_channels = set()
    for path in conv_data[path_column]:
        unique_channels.update(path.split(' > '))
    
    channel_dict = {channel: 0 for channel in sorted(unique_channels)}
    
    for _, row in conv_data.iterrows():
        path_list = row[path_column].split(' > ')
        num_channels = len(path_list)
        attribution_value = row[conversion_value_column] / num_channels
        
        for channel in path_list:
            channel_dict[channel] += attribution_value
    
    return channel_dict

def position_based_attribution(conv_data: pd.DataFrame, path_column: str, conversion_value_column: str, first_touch_pct: float = 0.4, last_touch_pct: float = 0.4) -> dict:
    """
    Calculate position-based attribution for each channel in the conversion paths.
    """
    if first_touch_pct + last_touch_pct > 1:
        raise ValueError("The sum of first_touch_pct and last_touch_pct must be less than or equal to 1.")
    
    unique_channels = set()
    for path in conv_data[path_column]:
        unique_channels.update(path.split(' > '))
    
    channel_dict = {channel: 0 for channel in sorted(unique_channels)}
    
    for _, row in conv_data.iterrows():
        path_list = row[path_column].split(' > ')
        num_channels = len(path_list)
        
        if num_channels == 1:
            channel_dict[path_list[0]] += row[conversion_value_column]
        else:
            first_touch_value = row[conversion_value_column] * first_touch_pct
            last_touch_value = row[conversion_value_column] * last_touch_pct
            middle_touch_value = row[conversion_value_column] * (1 - first_touch_pct - last_touch_pct)
            
            channel_dict[path_list[0]] += first_touch_value
            channel_dict[path_list[-1]] += last_touch_value
            
            if num_channels > 2:
                middle_value_per_channel = middle_touch_value / (num_channels - 2)
                for channel in path_list[1:-1]:
                    channel_dict[channel] += middle_value_per_channel
            else:
                channel_dict[path_list[0]] += middle_touch_value/2
                channel_dict[path_list[-1]] += middle_touch_value/2

    
    return channel_dict

def last_touch_attribution(conv_data: pd.DataFrame, path_column: str, conversion_value_column: str) -> dict:
    """
    Calculate last touch attribution for each channel in the conversion paths.
    """
    unique_channels = set()
    for path in conv_data[path_column]:
        unique_channels.update(path.split(' > '))
    
    channel_dict = {channel: 0 for channel in sorted(unique_channels)}
    
    for _, row in conv_data.iterrows():
        path_list = row[path_column].split(' > ')
        last_touch = path_list[-1]
        channel_dict[last_touch] += row[conversion_value_column]
    
    return channel_dict

def time_decay_attribution(conv_data: pd.DataFrame, path_column: str, conversion_value_column: str, exposure_time_column: str, lambda_: float = 0.5) -> dict:
    """
    Calculate time decay attribution for each channel in the conversion paths using an exponential decay function.
    """
    unique_channels = set()
    for path in conv_data[path_column]:
        unique_channels.update(path.split(' > '))
    
    channel_dict = {channel: 0 for channel in sorted(unique_channels)}
    
    for _, row in conv_data.iterrows():
        path_list = row[path_column].split(' > ')
        exposure_times = row[exposure_time_column]
        
        weights = np.exp(-lambda_ * np.array(exposure_times))
        total_weight = weights.sum()
        
        for channel, weight in zip(path_list, weights):
            channel_dict[channel] += (row[conversion_value_column] * weight / total_weight)
    
    return channel_dict

def pairs(lst):
    """Generate pairs of consecutive elements."""
    it1, it2 = tee(lst)
    next(it2, None)
    return zip(it1, it2)

def count_pairs(data: pd.DataFrame, path_column: str, conversion_column: str) -> defaultdict:
    """
    Count pairs of consecutive elements in paths.
    """
    counts = defaultdict(int)
    for row in data.itertuples():
        for ch_pair in pairs(['start'] + row.path.split(' > ')):
            counts[ch_pair] += 1
        if getattr(row, conversion_column) == 1:
            counts[(row.path.split(' > ')[-1], 'conversion')] += 1
        else:
            counts[(row.path.split(' > ')[-1], 'null')] += 1
    return counts

def transition_matrix(data: pd.DataFrame, path_column: str, conversion_column: str) -> defaultdict:
    """
    Calculate the transition matrix from the data.
    """
    tr = defaultdict(float)
    outs = defaultdict(int)
    pair_counts = count_pairs(data, path_column, conversion_column)
    for pair in pair_counts:
        outs[pair[0]] += pair_counts[pair]
    for pair in pair_counts:
        tr[pair] = pair_counts[pair] / outs[pair[0]]
    return tr

def prob_convert(data: pd.DataFrame, trans_mat: defaultdict, path_column: str, conversion_column: str, drop: str = None) -> float:
    """
    Calculate the probability of conversion with or without a specific channel.
    """
    if drop:
        filtered_data = data[data[path_column].apply(lambda x: drop not in x) & (data[conversion_column] > 0)]
    else:
        filtered_data = data[data[conversion_column] > 0]
    
    total_prob = 0
    for row in filtered_data.itertuples():
        path_probabilities = [trans_mat.get(pair, 0) for pair in pairs(['start'] + row.path.split(' > ') + ['conversion'])]
        path_probability = reduce(mul, path_probabilities, 1)
        total_prob += path_probability
    
    return total_prob

def calculate_removal_effect(data: pd.DataFrame, trans_mat: defaultdict, path_column: str, conversion_column: str) -> dict:
    """
    Calculate the effect of removing each channel on the overall conversion probability.
    """
    channels = set(data[path_column].str.split(' > ').sum()) - {'start', 'conversion', 'null'}
    removal_effect = {}
    overall_conversion_prob = prob_convert(data, trans_mat, path_column, conversion_column)
    
    for channel in channels:
        dropped_conversion_prob = prob_convert(data, trans_mat, path_column, conversion_column, drop=channel)
        removal_effect[channel] = (overall_conversion_prob - dropped_conversion_prob) / overall_conversion_prob
    
    total_effect = sum(removal_effect.values())
    normalized_effect = {channel: effect / total_effect for channel, effect in removal_effect.items()}
    
    return normalized_effect

def calculate_channel_values(data: pd.DataFrame, normalized_attribution: dict, path_column: str, value_column: str) -> dict:
    """
    Calculate the attribution value for each channel.
    """
    channel_values = defaultdict(float)
    total_conversion_value = data[data['conversion'] > 0][value_column].sum()
    
    for row in data.itertuples():
        path = row.path.split(' > ')
        path_value = row.conversion_value
        path_attribution_sum = sum(normalized_attribution[channel] for channel in path if channel in normalized_attribution)
        
        for channel in path:
            if channel in normalized_attribution:
                channel_values[channel] += path_value * (normalized_attribution[channel] / path_attribution_sum)
    
    normalization_factor = total_conversion_value / sum(channel_values.values())
    channel_values = {k: v * normalization_factor for k, v in channel_values.items()}
    return channel_values

def calculate_attribution_table(data, first_touch_pct = 0.4, last_touch_pct=0.4, time_decay_lambda = 0.5):
    # file_path = './mta/data/data.csv.gz'
    multi_channel_data, conv_multi_channel_data = process_data(data)
    
    linear_attr_dict = linear_attribution(conv_multi_channel_data, 'path', 'conversion_value')
    position_attr_dict = position_based_attribution(conv_multi_channel_data, 'path', 'conversion_value', first_touch_pct = first_touch_pct, last_touch_pct = last_touch_pct)
    last_touch_attr_dict = last_touch_attribution(conv_multi_channel_data, 'path', 'conversion_value')
    time_decay_attr_dict = time_decay_attribution(conv_multi_channel_data, 'path', 'conversion_value', 'ts', lambda_= time_decay_lambda)

    tr_matrix = transition_matrix(multi_channel_data, 'path', 'conversion')
    
    aggregated_conv_data = multi_channel_data[multi_channel_data['conversion'] > 0].groupby(['path', 'conversion']).agg({'conversion_value': 'sum'}).reset_index()
    overall_conversion_prob = prob_convert(aggregated_conv_data, tr_matrix, 'path', 'conversion')
    
    normalized_attribution = calculate_removal_effect(aggregated_conv_data, tr_matrix, 'path', 'conversion')
    markov_chain_attr_dict = calculate_channel_values(aggregated_conv_data, normalized_attribution, 'path', 'conversion_value')

    combined_dict = {
    # 'Linear': linear_attr_dict,
    'Position-Based': position_attr_dict,
    'Last Touch': last_touch_attr_dict,
    'Time Decay': time_decay_attr_dict,
    'Markov Chain': markov_chain_attr_dict}

    # Convert the combined dictionary to a DataFrame
    combined_df = pd.DataFrame(combined_dict).reset_index().rename(columns={'index': 'Channel'})
    return combined_df