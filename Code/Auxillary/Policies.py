from Networks import Actor, Critic, Context
import copy
import torch
import torch.nn.functional as F

class TD3_Policy:
    def __init__(
            self,
            state_dim,
            action_dim,
            actor_hidden,
            critic_hidden,
            context_hidden,
            actor_lr,
            critic_lr,
            exploration_noise,
            exploration_noise_clip,
            gamma,
            tau,
            policy_noise,
            policy_noise_clip,
            policy_freq,
            use_context=True,
            context_size=0,
            context_input_size=1000,
    ):
        self.max_action = 1
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.policy_noise_clip = policy_noise_clip
        self.exploration_noise_clip = exploration_noise_clip
        self.use_context = use_context
        self.context_input_size = context_input_size

        if self.use_context:
            self.context = Context(context_hidden, input_dim=11, output_dim=context_size)
            self.context_target = copy.deepcopy(self.context)

        self.actor = Actor(actor_hidden, state_dim, action_dim, self.max_action, context_size)
        self.critic = Critic(critic_hidden, state_dim, action_dim, context_size)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_target = copy.deepcopy(self.critic)

        if self.use_context:
            net = [self.context, self.critic]
            #net = [self.critic]
        else:
            net = [self.critic]

        parameters = set()
        for net_ in net:
            parameters |= set(net_.parameters())
        self.critic_optimizer = torch.optim.Adam(parameters, lr=critic_lr)

    def act(self,observation, transitions, det=False):
        with torch.no_grad():
            if self.use_context:
                context = self.context(transitions.float())
            else:
                context = None
            actions = self.actor(observation, context)
            if det:
                noise = torch.zeros(actions.shape)
            else:
                noise = (torch.randn_like(actions) * self.exploration_noise)
                div = noise /  self.exploration_noise_clip
                signs = torch.sign(div)
                dif = signs*div-torch.floor(signs*div)
                noise = signs*dif*self.exploration_noise_clip
            actions = (actions + noise).clamp(-self.max_action,self.max_action)
            #actions = (actions + noise)
        return actions.numpy()[0]

    def update(self, replay_buffer, batch_size, num_updates, task_id=None):
        for i in range(num_updates):
            observations, next_observations, actions, rewards, ids = replay_buffer.sample(batch_size, task_id)
            observations = torch.FloatTensor(observations)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_observations = torch.FloatTensor(next_observations)

            if self.use_context:
                transitions = replay_buffer.sample(self.context_input_size, ids[0])
                transitions = torch.FloatTensor(transitions)
                transitions = torch.unsqueeze(transitions, 0)
                for env_id in range(len(ids)-1):
                    samples_id = replay_buffer.sample(self.context_input_size, ids[env_id+1])
                    samples_id = torch.FloatTensor(samples_id)
                    samples_id = torch.unsqueeze(samples_id, 0)
                    transitions = torch.cat((transitions, samples_id), dim=0)
                context = self.context(transitions.float())
                context_target = self.context_target(transitions.float()).detach()
            else:
                context = None
                context_target = None

            with torch.no_grad():
                #noise = (torch.randn_like(actions) * self.policy_noise) \
                #    .clamp(-self.policy_noise_clip, self.policy_noise_clip)
                noise = (torch.randn_like(actions) * self.policy_noise)
                div = noise /  self.policy_noise_clip
                signs = torch.sign(div)
                dif = signs*div-torch.floor(signs*div)
                noise = signs*dif*self.policy_noise_clip

                next_actions = (self.actor_target(next_observations, context_target)\
                                + noise).clamp(-self.max_action, self.max_action)
                target_Q1, target_Q2 = self.critic_target(next_observations, next_actions, context_target)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + self.gamma * target_Q


            current_Q1, current_Q2 = self.critic(observations, actions, context)
            critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
            critic_loss = critic_loss.mean()
            self.critic_optimizer.zero_grad()
            critic_loss.double().backward()
            #print(critic_loss)
            self.critic_optimizer.step()

            if i % self.policy_freq == 0:
                if self.use_context:
                    context = context.detach()
                actor_loss = -self.critic(observations, self.actor(observations, context), context, which='Q1')
                actor_loss = actor_loss.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                if self.use_context:
                    for param, target_param in zip(self.context.parameters(), self.context_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_networks(self, path):
        torch.save(self.critic.state_dict(), path / "critic.pt")
        torch.save(self.critic_target.state_dict(), path / "critic_target.pt")
        torch.save(self.critic_optimizer.state_dict(), path / "critic_optimizer.pt")

        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.actor_target.state_dict(), path / "actor_target.pt")
        torch.save(self.actor_optimizer.state_dict(), path / "actor_optimizer.pt")

        if self.use_context:
            torch.save(self.context.state_dict(), path / "context.pt")

    def load_networks(self, path, target=False):
        self.critic.load_state_dict(torch.load(path / "critic.pt"))
        #for param in self.critic.parameters():
        #    print(param.data)
        #self.critic_optimizer.load_state_dict(torch.load(path / "critic_optimizer.pt"))

        self.actor.load_state_dict(torch.load(path / "actor.pt"))
        #self.actor_optimizer.load_state_dict(torch.load(path / "actor_optimizer.pt"))

        for param in self.actor.parameters():
            print(param.data)


        if self.use_context:
            self.context.load_state_dict(torch.load(path / "context.pt"))


        if target:
            self.critic_target.load_state_dict(torch.load(path / "critic_target.pt"))
            self.actor_target.load_state_dict(torch.load(path / "actor_target.pt"))
        else:
            self.critic_target = copy.deepcopy(self.critic)
            self.actor_target = copy.deepcopy(self.actor)


















