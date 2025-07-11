from typing import Dict, List, Optional, Tuple

from swift.llm.template.utils import split_parts_by_regex, split_str_parts_by


def calculate_loss_scale(query: str,
                         response: str,
                         response_loss_scale_map: Dict[str, list],
                         query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```
    Returns:
        A tuple of agent response parts and their weights.
    """
    # query loss scale map
    if query_loss_scale_map is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]
    delimiters = list(k for k in response_loss_scale_map.keys() if len(response_loss_scale_map[k]) == 2)
    agent_parts = split_str_parts_by(response, delimiters)
    regex_delimiters = {k: v for k, v in response_loss_scale_map.items() if len(v) == 1}
    if len(regex_delimiters):
        split_parts_by_regex(agent_parts, regex_delimiters)
    weights = []
    agent_content = []
    for c in agent_parts:
        if isinstance(c['key'], (float, int)):
            weights += [c['key']]
            agent_content.append(c['content'])
        else:
            if c['key'] in response_loss_scale_map:
                weights += [response_loss_scale_map[c['key']][0]]
                weights += [response_loss_scale_map[c['key']][1]]
                agent_content.append(c['key'])
                agent_content.append(c['content'])
            else:
                weights += [1.0]
                agent_content.append(c['content'])
    return agent_content, weights
