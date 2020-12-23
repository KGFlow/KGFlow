import numpy as np


def entity_negative_sampling(source_entities, relations, kg, target_entity_type="tail", filtered=False):
    """

    :param source_entities:
    :param relations:
    :param kg: KG
    :param target_entity_type: "head" | "tail"
    :param filtered:
    :return:
    """

    if not filtered:
        return np.random.choice(kg.num_entities, len(source_entities), replace=False)

    if target_entity_type == "tail":
        source_relation_target_dict = kg.head_relation_tail_dict
    else:
        source_relation_target_dict = kg.tail_relation_head_dict

    neg_entities = []
    for source, relation in zip(source_entities, relations):
        pos_entities = source_relation_target_dict[int(source)][int(relation)]
        while True:
            neg_entity = np.random.randint(0, kg.num_entities)
            if neg_entity not in pos_entities:
                neg_entities.append(neg_entity)
                break

    return np.array(neg_entities)
