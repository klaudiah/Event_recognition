import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import netgraph


def map_node_color(node_data):
    if 'user' in node_data:
        return 'red'
    elif 'event' in node_data:
        return 'green'
    return 'black'


def prepare_user_dict(user):
    return {
        'user': True,
        'locale': user['locale'],
        'birthyear': int(user['birthyear']),
        'gender': user['gender'],
        'joinedAt': user['joinedAt'],
        'location': user['location'],
        'timezone': int(user['timezone'])
    }


train = pd.read_csv("..//dataset//train.csv")
events = pd.read_csv("..//dataset//events.csv")
users = pd.read_csv("..//dataset//users.csv")

user_node_attrs = {
    'user': True,
    'locale': None,
    'birthyear': None,
    'gender': None,
    'joinedAt': None,
    'location': None,
    'timezone': None
}

event_node_attrs = {
    'event': True,

}

network = nx.Graph()

for id, row in train.iterrows():
    if not network.has_node(row['user']):
        network.add_node(row['user'], **user_node_attrs)
    if not network.has_node(row['event']):
        network.add_node(row['event'], **event_node_attrs)
    if not network.has_edge(row['user'], row['event']):
        network.add_edge(row['user'], row['event'],
                         timestamp=row['timestamp'],
                         invited=row['invited'],
                         interested=row['interested'],
                         not_interested=row['not_interested'])

draw_options = {
    'node_color': 'red',
    'node_size': 5,
}

node_colors = [map_node_color(node[1]) for node in network.nodes(data=True)]

#plt.subplot(242)
nx.draw_networkx(network, with_labels=True, node_color=node_colors)
#netgraph.draw(network, with_labels=True)
#plot_instance = netgraph.InteractiveGraph(network)
plt.show()
