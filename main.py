import functools

import haiku
import jax
import jraph
import optax
from jax import numpy


@jraph.concatenated_args
def edge_update_fn(feats: numpy.ndarray) -> numpy.ndarray:
    net = haiku.Sequential(
        [
            haiku.Linear(8),
            jax.nn.relu,
            haiku.Linear(8),
        ]
    )
    return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: numpy.ndarray) -> numpy.ndarray:
    net = haiku.Sequential(
        [
            haiku.Linear(8),
            jax.nn.relu,
            haiku.Linear(8),
        ]
    )
    return net(feats)


@jraph.concatenated_args
def global_update_fn(feats: numpy.ndarray) -> numpy.ndarray:
    net = haiku.Sequential(
        [
            haiku.Linear(8),
            jax.nn.relu,
            haiku.Linear(2),
        ]
    )
    return net(feats)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    graph = graph._replace(globals=numpy.zeros([graph.n_node.shape[0], 1]))
    embed = jraph.GraphMapFeatures(
        haiku.Linear(8),
        haiku.Linear(8),
        haiku.Linear(8),
    )
    update = jraph.GraphNetwork(
        edge_update_fn,
        node_update_fn,
        global_update_fn,
    )
    return update(embed(graph))


def compute_loss(params, graph: jraph.GraphsTuple, label, net):
    pred_graph: jraph.GraphsTuple = net.apply(params, graph)
    pred_nodes = pred_graph.nodes
    a = pred_nodes[0] @ pred_nodes[1]
    b = pred_nodes[0] @ pred_nodes[2]
    c = pred_nodes[1] @ pred_nodes[2]
    loss = (a - 1) ** 2 + b ** 2 + c ** 2
    return loss, -loss


# features for nodes
nodes = numpy.array([[0.0], [1.0], [2.0]])

# edges
senders = numpy.array([0, 1, 2])
receivers = numpy.array([1, 2, 0])

# features for edges
edges = numpy.array([[5.0], [6.0], [7.0]])

n_node = numpy.array([3])
n_edge = numpy.array([3])

global_context = numpy.array([[1]])
graph = jraph.GraphsTuple(
    nodes=nodes,
    senders=senders,
    receivers=receivers,
    edges=edges,
    n_node=n_node,
    n_edge=n_edge,
    globals=global_context,
)


net = haiku.without_apply_rng(haiku.transform(net_fn))
params = net.init(jax.random.PRNGKey(42), graph)
opt_init, opt_update = optax.adam(1e-4)
opt_state = opt_init(params)

# compute_loss(params, graph, None, net)
# exit()

compute_loss_fn = functools.partial(compute_loss, net=net)
compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

for _ in range(20):
    label = None
    (loss, _metric), grad = compute_loss_fn(params, graph, label)
    print("loss =", loss)
    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    x = net.apply(params, graph)
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"- {i} @ {j} = {x.nodes[i] @ x.nodes[j]}")
