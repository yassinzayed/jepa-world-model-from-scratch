"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    # TODO: Initialize a random agent position state inside a square 2D room...
    if seed is None:
        generator = None
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
    current_state = torch.randint(low=0, high=room_size, size=(2,), generator=generator, dtype=torch.float32)
    return current_state

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    # TODO: Apply a discrete action to the agent state with wall clamping.
    if action == 0:
        state_new = torch.stack([state[0], state[1]-1])
    elif action == 1:
        state_new = torch.stack([state[0], state[1]+1])
    elif action == 2:
        state_new = torch.stack([state[0]-1, state[1]])
    elif action == 3:
        state_new = torch.stack([state[0]+1, state[1]])
    return torch.clamp(state_new, 0, room_size - 1)

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    # TODO: Render a (1, room_size, room_size) float32 obs with agent pixel 1.0...
    rendering = torch.zeros((1, room_size, room_size))
    rendering[0, int(state[1]), int(state[0])] = 1.0
    return rendering

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Reset the environment by sampling a new state and its observation.
    state = init_env_state(room_size, seed)
    observation = render_observation(state, room_size)
    return (state, observation)

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Advance the env by one action; return (next_state, next_observation).
    next_state = apply_action(state, action, room_size)
    next_observation = render_observation(next_state, room_size)
    return (next_state, next_observation)

# Step 6 - collect_random_transitions
def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: collect a dataset of (obs, action, next_obs, state, next_state) transitions...
    generator = torch.Generator().manual_seed(seed)
    state, observation = env_reset(room_size, seed=seed)
    if num_transitions == 0:
        return {
            "observations": observation.unsqueeze(0)[:0],
            "actions": torch.empty((0,), dtype=torch.int64),
            "next_observations": observation.unsqueeze(0)[:0],
            "states": state.unsqueeze(0)[:0],
            "next_states": state.unsqueeze(0)[:0]
        }
    observations, actions, next_observations, states, next_states = (
        [],
        [],
        [],
        [],
        []
    )
    transitions = torch.randint(0, 4, (num_transitions,), generator=generator)
    for i in transitions:
        next_state, next_observation = env_step(state=state, action=int(i.item()), room_size=room_size)
        observations.append(observation)
        next_observations.append(next_observation)
        states.append(state)
        next_states.append(next_state)
        state = next_state
        observation = next_observation
    return {
        'observations': torch.stack(observations),
        'actions': transitions,
        'next_observations': torch.stack(next_observations),
        'states': torch.stack(states),
        'next_states': torch.stack(next_states)
    }

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: Build a JEPA training-ready transition dataset by collecting random transitions...
    return collect_random_transitions(num_transitions, room_size, seed)

# Step 8 - init_encoder_params
def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    # TODO: Initialize parameters of a small CNN encoder that maps pixel observations to latent embeddings.
    torch.manual_seed(seed)
    h1 = (room_size-1)//2 + 1
    h2 = (h1-1)//2 + 1
    fc_in = 32*(h1**2)
    shape_w1 = (16, obs_channels, 3, 3)
    shape_w2 = (32, 16, 3, 3)
    shape_fc = (latent_dim, fc_in)
    return {
        'conv1_w': torch.randn(shape_w1, requires_grad=True) * 0.1,
        'conv1_b': torch.zeros(16, requires_grad=True),
        'conv2_w': torch.randn(shape_w2, requires_grad=True) * 0.1,
        'conv2_b': torch.zeros(32, requires_grad=True),
        'fc_w': torch.randn(shape_fc, requires_grad=True) * 0.1,
        'fc_b': torch.zeros(latent_dim, requires_grad=True)
    }

# Step 9 - encoder_forward
def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Run a small two-layer CNN that maps pixel observations to latent embeddings.
    res1 = torch.nn.functional.relu(torch.nn.functional.conv2d(obs, weight=encoder_params['conv1_w'], bias=encoder_params['conv1_b'], stride=1, padding=1))
    res2 = torch.nn.functional.relu(torch.nn.functional.conv2d(res1, weight=encoder_params['conv2_w'], bias=encoder_params['conv2_b'], stride=2, padding=1)).flatten(1)
    res = torch.nn.functional.linear(res2, weight=encoder_params['fc_w'], bias=encoder_params['fc_b'])
    return res

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    # TODO: Create the EMA target encoder by deep-copying the online encoder params.
    if not(encoder_params):
        return {}
    else:
        return {
            'conv1_w': encoder_params['conv1_w'].detach().clone(),
            'conv1_b': encoder_params['conv1_b'].detach().clone(),
            'conv2_w': encoder_params['conv2_w'].detach().clone(),
            'conv2_b': encoder_params['conv2_b'].detach().clone(),
            'fc_w': encoder_params['fc_w'].detach().clone(),
            'fc_b':encoder_params['fc_b'].detach().clone()
        }

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    # TODO: Refresh target encoder params via EMA of the online encoder.
    result_dict = {}
    for key in target_params:
        current_res = tau*target_params[key] + (1-tau)*encoder_params[key]
        result_dict[key] = current_res
    return result_dict

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Batch-encode observations into latent embeddings using encoder params.
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    # TODO: Initialize and return a dict of MLP predictor parameters.
    torch.manual_seed(seed)
    action_embed_w = torch.randn((action_dim, latent_dim))*0.02
    fc1_w = torch.randn((hidden_dim, 2*latent_dim))*0.02
    fc2_w = torch.randn((latent_dim, hidden_dim))*0.02
    fc1_b = torch.zeros((hidden_dim,), requires_grad=True)
    fc2_b = torch.zeros((latent_dim,), requires_grad=True)
    return {
        'action_embed_w': action_embed_w.requires_grad_(True),
        'fc1_w': fc1_w.requires_grad_(True),
        'fc1_b': fc1_b,
        'fc2_w': fc2_w.requires_grad_(True),
        'fc2_b': fc2_b
    }

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Embed discrete actions into continuous vectors via a learned action embedding table.
    weight_matrix = predictor_params['action_embed_w']
    return weight_matrix[actions]

# Step 15 - predictor_forward
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Implement predictor_forward, the forward pass of the action-conditioned dynamics predictor.
    action_embeddings = embed_action(actions, predictor_params)
    concat = torch.cat((embeddings, action_embeddings), dim=-1)
    res1 = torch.relu(torch.nn.functional.linear(concat, predictor_params['fc1_w'], predictor_params['fc1_b']))
    res2 = torch.nn.functional.linear(res1, predictor_params['fc2_w'], predictor_params['fc2_b'])
    return res2

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Predict the next latent embedding given current embeddings and actions...
    return predictor_forward(embeddings, actions, predictor_params)

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: Compute the JEPA prediction loss as mean squared error...
    residual = (predicted-target)**2
    return torch.mean(residual)

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # TODO: Compute VICReg variance hinge loss encouraging each dim std >= gamma.
    std_dim = torch.sqrt(torch.var(embeddings, dim=0)+eps)
    loss = torch.mean(torch.relu(gamma-std_dim))
    return loss

# Step 19 - covariance_loss
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Implement `covariance_loss` to compute the VICReg covariance regularization from embeddings.
    z_centered = embeddings-torch.mean(embeddings, dim=0)
    covariance = ((z_centered.T @ z_centered)/(embeddings.shape[0]-1))
    L = (1/embeddings.shape[1])*(covariance.pow(2).sum()-covariance.diagonal().pow(2).sum())
    return L

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    # TODO: Combine variance and covariance terms into the VICReg regularizer...
    return var_weight*variance_loss(embeddings, gamma) + cov_weight*covariance_loss(embeddings)

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    # TODO: Compose the full JEPA objective from prediction and VICReg terms...
    return pred_weight*prediction_loss(predicted, target) + vicreg_regularizer(online_embeddings, var_weight, cov_weight)

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Measure collapse as the mean of per-dimension batch stds
    return embeddings.std(dim=0).mean()

# Step 23 - jepa_training_step
def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    # TODO: Perform one full JEPA gradient update on a batch of transitions
    encoder_params = {
        k: p.detach().requires_grad_(True)
        for k, p in encoder_params.items()
    }
    predictor_params = {
        k: p.detach().requires_grad_(True)
        for k, p in predictor_params.items()
    }
    online_enc = encode_batch(batch['observations'], encoder_params)
    with torch.no_grad():
        target = encode_batch(batch['next_observations'], target_params)
    predicted = predict_next_embedding(online_enc, batch['actions'], predictor_params)
    loss = jepa_loss(predicted, target, online_enc)
    loss.backward()
        # SGD updates
    updated_encoder = {
        k: (p - lr * p.grad).detach().requires_grad_(True)
        if p.grad is not None else p.detach().requires_grad_(True)
        for k, p in encoder_params.items()
    }
    updated_predictor = {
        k: (p - lr * p.grad).detach().requires_grad_(True)
        if p.grad is not None else p.detach().requires_grad_(True)
        for k, p in predictor_params.items()
    }
    target_params = ema_update(target_params, updated_encoder, tau)
    updated_target = ema_update(target_params, updated_encoder, tau)
    collapse_value = collapse_metric(online_enc.detach()).item()
    return (
        updated_encoder,
        updated_target,
        updated_predictor,
        loss.item(),
        collapse_value
    )

# Step 24 - train_jepa
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    # TODO: Train JEPA for num_steps with collapse monitoring, return params + history
    torch.manual_seed(seed)
    n = dataset['observations'].shape[0]
    history = []
    for i in range(num_steps):
        indices = torch.randint(0, n, (batch_size,))
        batch = {
            'observations': dataset['observations'][indices],
            'actions': dataset['actions'][indices],
            'next_observations': dataset['next_observations'][indices]
        }
        encoder_params, target_params, predictor_params, loss_value, collapse_value = jepa_training_step(batch, encoder_params, target_params, predictor_params, lr, tau)
        history.append({'loss': loss_value, 'collapse': collapse_value})
    return (encoder_params, target_params, predictor_params, history)

# Step 25 - rollout_latent_dynamics (not yet solved)
# TODO: implement

# Step 26 - multi_step_prediction_error (not yet solved)
# TODO: implement

# Step 27 - init_linear_probe
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    # TODO: Initialize a linear probe that maps latent embeddings to true agent state (x, y).
    torch.manual_seed(seed)
    weights = torch.randn(state_dim, latent_dim)*0.01
    bias = torch.zeros(state_dim)
    return {
        'w': weights,
        'b': bias
    }

# Step 28 - train_linear_probe
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    # TODO: Train the linear probe via MSE regression from frozen embeddings to true agent states.
    w, b = probe_params['w'].clone().detach().requires_grad_(True), probe_params['b'].clone().detach().requires_grad_(True)
    for i in range(num_steps):
        pred = embeddings @ w + b
        loss = torch.mean((pred-states)**2)
        loss.backward()
        with torch.no_grad():
            w -= lr*w.grad
            b -= lr*b.grad
        w.grad.zero_()
        b.grad.zero_()
    return {
        'w': w.detach().clone(),
        'b': b.detach().clone()
    }

# Step 29 - probe_state_recovery (not yet solved)
# TODO: implement

# Step 30 - encode_goal
def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    # TODO: Convert a desired agent position into a latent goal embedding.
    observation = render_observation(goal_state, room_size)
    observation.unsqueeze(0)
    observation = encoder_forward(observation, encoder_params)
    return observation.squeeze(0)

# Step 31 - latent_cost (not yet solved)
# TODO: implement

# Step 32 - sample_action_sequences (not yet solved)
# TODO: implement

# Step 33 - score_action_sequences (not yet solved)
# TODO: implement

# Step 34 - select_best_plan (not yet solved)
# TODO: implement

# Step 35 - mpc_step (not yet solved)
# TODO: implement

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

