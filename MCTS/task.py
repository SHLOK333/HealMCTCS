import random
from tasks.science import SearchTask
from MCTS.base import treeNode
from models.get_response import *
from MCTS.mcts import MCTS
from utils.verify_MATH import exact_match_score, grade_answer, extract_answer
from utils.verify_llm import llm_verify
from utils.solution_summary_extractor import extract_summary_from_solution


class MCTS_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='en', weighted_verify=False):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'prm' if USE_PRM else 'vm'
        self.lang = 'en'
        self.weighted_verify = weighted_verify

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

    def get_next_step(self, y, step_n):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        else:
            if self.propose_method == 'gpt':
                prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
            elif self.propose_method == 'mistral' or self.propose_method == 'llama':
                prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)
            else:
                prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('Failed to get next step!\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if "Next step:" in p:
            stp = p.split('Next step:')[1].strip()
            if len(stp) < 2:
                print('Step output too short!\n')
                return ''
            if stp in y:
                print('Step output repeated!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + stp
            print(f'Normalized new step:{revised_}\n')
            return revised_ + '\n'

        elif "Step" in p and ":" in p:
            pre_len = len(p.split(':')[0])
            p_ = p[pre_len:]
            p_ = p_.split('Step')[0].strip()
            if len(p_) < 4:
                print('Step output too short!\n')
                return ''
            p_ = p_[1:].strip()
            if p_ in y:
                print('Step output repeated!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'Normalized new step:{revised_}\n')
            return revised_ + '\n'

        else:
            p_ = p.strip()
            if len(p_) < 3:
                print('Step output too short!\n')
                return ''
            if p_ in y:
                print('Step output repeated!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'Normalized new step:{revised_}\n')
            return revised_ + '\n'

    def get_next_step_use_reflection(self, y, step_n, reflection):
        if self.propose_method == 'gpt' or self.propose_method == 'local':
            propose_prompt = self.zero_single_propose_wrap_use_reflection_gpt(self.question, y, step_n, reflection,
                                                                              self.lang)
        else:
            propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection,
                                                                          self.lang)
        response = get_proposal(propose_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('Failed to get next step!\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if "Next step:" in p:
            stp = p.split('Next step:')[1].strip()
            if len(stp) < 2:
                print('Step output too short!\n')
                return ''
            if stp in y:
                print('Step output repeated!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + stp
            print(f'Normalized new step:{revised_}\n')
            return revised_ + '\n'

        elif "Step" in p and ":" in p:
            pre_len = len(p.split(':')[0])
            p_ = p[pre_len:]
            p_ = p_.split('Step')[0].strip()
            if len(p_) < 4:
                print('Step output too short!\n')
                return ''
            p_ = p_[1:].strip()
            if p_ in y:
                print('Step output repeated!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'Normalized new step:{revised_}\n')
            return revised_ + '\n'

        else:
            print('Output format error!\n')
            return ''

    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_simple_mistral(self.question, y, step_n)
        else:
            reflection_prompt = self.single_reflection_wrap_simple(self.question, y, step_n, self.lang)
        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            cnt -= 1
        if not response:
            print('Failed to get reflection!\n')
            return '<end>'

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if 'unsolved' in p or step_n <= 1:
            print('Normalized reflection: <continue>\n')
            return '<continue>'
        elif 'solved' in p:
            print('Normalized reflection: <end>\n')
            return '<end>'
        else:
            print('Normalized reflection: <continue>\n')
            return '<continue>'

    def get_reflection(self, y, step_n):
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)

        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, self.max_new_tokens)
            cnt -= 1
        if not response:
            print('Failed to get reflection!\n')
            return ''

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if 'Problem solved' in p:
            print('Normalized reflection: <end>\n')
            return '<end>'
        else:
            if 'Analysis:' not in p:
                print('Output format error!\n')
                return ''
            revised_ = p.split('Analysis:')[1].strip()
            print(f'Normalized reflection:{revised_}\n')
            return revised_

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        if self.value_method == 'local':
            prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y
            value = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
                              self.max_length, self.low, self.high)
            print(f'Got value:{value}\n')
            self.value_cache.update({y: value})
            return value

        else:
            prompt = self.value_prompt_wrap(self.question, y)
            response = get_value(prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
                                 self.max_length, self.low, self.high)
            value = self.value_outputs_unwrap(response, self.low, self.high)
            print(f'Got value:{value}\n')
            self.value_cache.update({y: value})
            return value

    def get_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, 128)
        if not response:
            print('Failed to get summary!\n')
            return ''
        p = ''
        for _ in response:
            p = p + _
        summ = p.strip()
        print(f'Got summary:{summ}\n')
        return summ

    def get_MATH_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, 128)
        if not response:
            print('Failed to get summary!\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        print(f'Got summary:{p}\n')
        return p

    def verify_end_nodes(self, root):
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()
        flag = False
        for leaf in end_leaf_nodes:
            leaf.on_final_route = True
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(leaf.y)
                else:
                    summ = self.get_summary(leaf.y)
                if summ:
                    leaf.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(leaf.y)
                leaf.summary = summ

            if self.verify_method == 'string':
                result = exact_match_score(summ, self.answer)
            else:
                result = llm_verify(summ, self.answer)
            if result:
                if self.reward_model_type == 'vm':
                    leaf.min_steps_to_correct = 1
                else:
                    leaf.he = 1
                flag = True
        return flag, end_leaf_nodes

    def get_final_solution(self, root, weighted):
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()

        if not end_leaf_nodes or not weighted:
            if not end_leaf_nodes:
                best_node, best_V = root.getBestV()
            else:
                sorted_nodes = sorted(end_leaf_nodes, key=lambda x: x.V, reverse=True)
                best_node = sorted_nodes[0]
            solution = best_node.y
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(solution)
                else:
                    summ = self.get_summary(solution)
                if summ:
                    best_node.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(solution)
                best_node.summary = summ
            return solution, summ

        else:
            all_answers = {}  # {answer: [solution, summ, value]}
            for leaf in end_leaf_nodes:
                cnt = 5
                summ = ''
                while cnt:
                    if self.verify_method == 'string':
                        summ = self.get_MATH_summary(leaf.y)
                    else:
                        summ = self.get_summary(leaf.y)
                    if summ:
                        leaf.summary = summ
                        break
                    else:
                        cnt -= 1
                if not summ:
                    summ = extract_summary_from_solution(leaf.y)
                    leaf.summary = summ

                extracted_answer = extract_answer(summ)
                if extracted_answer in all_answers.keys():
                    all_answers[extracted_answer][2] += leaf.V
                else:
                    all_answers[extracted_answer] = [leaf.y, summ, leaf.V]

            best_answer = max(all_answers.values(), key=lambda x: x[2])
            solution = best_answer[0]
            summ = best_answer[1]
            return solution, summ

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
        # vm
        if self.reward_model_type == 'vm':
            if self.sample_value != 'full':
                if self.evaluate == 'scibench':
                    solution = node.y
                    summary = self.get_summary(solution)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summary,
                                    'finish': finish}
                    if self.sample_value == 'simple':
                        node.trace_route()
                        new_value_samples = node.get_new_value_samples()
                        final_answer.update({'value_samples': new_value_samples})
                else:
                    solution = node.y
                    cnt = 5
                    summ = ''
                    while cnt:
                        if self.verify_method == 'string':
                            summ = self.get_MATH_summary(solution)
                        else:
                            summ = self.get_summary(solution)
                        if summ:
                            node.summary = summ
                            break
                        else:
                            cnt -= 1

                    if not summ:
                        summ = extract_summary_from_solution(solution)
                        node.summary = summ

                    result = exact_match_score(summ, self.answer)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'accurate': result, 'real_answer': self.answer}
                return final_answer, root
            else:
                if not self.evaluate:
                    assert self.answer is not None, 'Answer is None!\n'
                    flag, end_leaf_nodes = self.verify_end_nodes(root)

                    new_policy_samples = []
                    for leaf in end_leaf_nodes:
                        solution = leaf.y
                        summ = leaf.summary
                        correct = True if leaf.min_steps_to_correct == 1 else False
                        new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
                        new_policy_samples.append(new_policy_sample)

                    if flag:
                        new_value_samples = root.get_full_value_samples_vm(end_leaf_nodes)
                    else:
                        new_value_samples = []
                    final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
                                    'value_samples': new_value_samples, 'real_answer': self.answer}
                    return final_answer, root
                else:
                    assert self.answer is not None, 'Answer is None!\n'
                    solution, summ = self.get_final_solution(root, self.weighted_verify)
                    if not summ:
                        result = False
                    else:
                        result = exact_match_score(summ, self.answer)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'accurate': result, 'real_answer': self.answer}
                    return final_answer, root

        else:
            assert self.sample_value, 'Only sampling is supported for prm!\n'
            assert self.answer is not None, 'Answer is None!\n'
            flag, end_leaf_nodes = self.verify_end_nodes(root)

            new_policy_samples = []
            for leaf in end_leaf_nodes:
                solution = leaf.y
                summ = leaf.summary
                correct = True if leaf.he == 1 else False
                new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
                new_policy_samples.append(new_policy_sample)

            if flag:
                new_value_samples = root.get_full_value_samples_prm(end_leaf_nodes)
            else:
                new_value_samples = []
            final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
                            'value_samples': new_value_samples, 'real_answer': self.answer}
            return final_answer, root
