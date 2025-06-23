import argparse
import torch
from models import Transformer, LogTemplateMiner, LogTokenizer, load_config
from collections import deque
import os

try:
    import graphviz
except ImportError:
    graphviz = None

class PredictionNode:
    def __init__(self, event_id, prob, parent=None):
        self.event_id = event_id
        self.prob = prob
        self.parent = parent
        self.children = []

    def path(self):
        node, path = self, []
        while node:
            path.append(node.event_id)
            node = node.parent
        return list(reversed(path))

def recursive_topk_predict(model, template_miner, tokenizer, input_x, input_sequence, steps=3, topk=3, temperature=1.0):
    assert topk > 0, "topk needs to be bigger than 0!"

    def recurse(current_node, current_sequence, depth):
        if depth == 0:
            return [current_node]

        x_tensor = torch.tensor(current_node.path()).unsqueeze(0)
        sequence_tensor = torch.tensor([current_sequence])

        topk_preds = model.topk_predict(x_tensor, sequence_tensor, temperature=temperature, top_k=topk)

        leaf_nodes = []
        for event_id, prob in topk_preds:
            child = PredictionNode(event_id=event_id, prob=prob, parent=current_node)
            current_node.children.append(child)

            template = template_miner.decode_event_id_sequence(event_id)
            tokenized_template = tokenizer.transform(template)
            new_sequence = current_sequence[1:] + [tokenized_template]

            leaf_nodes.extend(recurse(child, new_sequence, depth - 1))
        return leaf_nodes

    root_event_id = input_x[-1].item() if isinstance(input_x, torch.Tensor) else input_x[-1]
    root = PredictionNode(event_id=root_event_id, prob=1.0)

    return recurse(root, input_sequence, steps), root

def write_predictions_to_file(results, tokenizer, output_file):
    with open(output_file, 'w') as f:
        for node in results:
            path = node.path()
            decoded = tokenizer.decode_event_id_sequence(path)
            f.write(f"Path: {decoded} — Cumulative Confidence: {node.prob:.6f}\n")
    print(f"Prediction results written to {output_file}")

def visualize_tree(root_node, tokenizer, output_path):
    if graphviz is None:
        print("Graphviz is not installed. Skipping tree rendering.")
        return

    dot = graphviz.Digraph(comment="Prediction Tree")

    queue = deque([root_node])
    id_counter = 0
    node_ids = {}

    while queue:
        node = queue.popleft()
        path = tokenizer.decode_event_id_sequence([node.event_id])
        label = f"{path} ({node.prob:.4f})"

        node_id = f"node{id_counter}"
        node_ids[node] = node_id
        dot.node(node_id, label)
        id_counter += 1

        if node.parent:
            dot.edge(node_ids[node.parent], node_id)

        for child in node.children:
            queue.append(child)

    dot.render(output_path, format='png', cleanup=True)
    print(f"Prediction tree visual saved as {output_path}.png")

def main():
    parser = argparse.ArgumentParser(description="Transformer Log Predictor")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--prediction_steps', type=int, default=1, help='Number of steps to predict')
    parser.add_argument('--branches', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='prediction_results.txt')
    parser.add_argument('--render_tree', type=str, help="Output PNG file path for prediction tree visualization")
    args = parser.parse_args()

    config = load_config(args.config_path)['base_config']

    template_miner = LogTemplateMiner(config['dataset']['drain_path'])
    template_miner.load_state()
    config['model']['vocab_size'] = template_miner.get_vocab_size()

    tokenizer = LogTokenizer(
        config['tokenizer']['tokenizer_length'],
        tokenizer_path=config['tokenizer']['tokenizer_path']
    )
    config['tokenizer']['vocab_size'] = tokenizer.get_vocab_size()

    model = Transformer(config)
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(config['device']).eval()

    import importlib
    dataset_module = importlib.import_module(f"dataset_class.{config['dataset']['class']}")
    dataset_class = getattr(dataset_module, "Dataset")
    dataset = dataset_class(
        path=config['dataset']['path'],
        context_length=config['model']['context_length'],
        template_miner=template_miner.transform,
        tokenizer=tokenizer.transform
    )

    data, sequences = dataset[0][0]

    print(f"\nRecursive Top-{args.branches} Prediction Tree (depth={args.prediction_steps}):")
    results, root_node = recursive_topk_predict(model, template_miner, tokenizer, data, sequences, steps=args.prediction_steps, topk=args.branches)

    results.sort(key=lambda node: node.prob, reverse=True)

    write_predictions_to_file(results, tokenizer, args.output_file)

    if args.render_tree:
        output_path = os.path.splitext(args.render_tree)[0]
        visualize_tree(root_node, tokenizer, output_path)

if __name__ == "__main__":
    main()