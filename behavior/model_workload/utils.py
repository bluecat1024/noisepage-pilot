import pandas as pd

def compute_frame(frame, deltas, pk_keys, all_keys):
    # FIXME(UPDATE): Assume UPDATEs do not effect the primary keys of the tuples that they change.
    # We actually make a larger statement: UPDATEs don't change ANY keys whatsoever and so
    # don't require index inserts if a HOT is determined to be valid. Else we need both the
    # "old" keys as DELETE and the "new" keys as INSERT.
    insert_index = deltas[deltas.is_insert].index
    delete_index = deltas[deltas.is_delete].index
    if len(insert_index) == 0 and len(delete_index) == 0:
        return frame, False

    insert_tuples = {}
    delete_tuples = []
    visit_index = insert_index.union(delete_index)
    idx = {name: i for i, name in enumerate(deltas.columns)}
    for tup in deltas.loc[visit_index].itertuples(index=False):
        key = tuple(tup[idx[pk]] for pk in pk_keys)
        if tup.is_insert:
            # FIXME(UNIQUE): We don't address two UNIQUE inserts directly.
            # We just assume that pk_keys can distinguish between referential INSERT/DELETE.
            if key in insert_tuples:
                insert_tuples[key].append(tup)
            else:
                insert_tuples[key] = [tup]
        else:
            assert tup.is_delete
            if key in insert_tuples:
                # FIXME(CONCURRENT): We assume every thread is a snowflake in its painted world.
                insert_tuples.pop(key, None)
            delete_tuples.append(key)

    if len(delete_tuples) > 0:
        # Set the is_valid flag to False and then we'll adjust the whole frame later together with inserts.
        # This is done for performance reasons.
        frame.set_index(keys=pk_keys, inplace=True)
        frame = frame[~frame.index.isin(delete_tuples)]
        frame.reset_index(drop=False, inplace=True)

    insert_tuples = [i for k, v in insert_tuples.items() for i in v]
    if len(insert_tuples) > 0:
        # Augment the frame with the newly inserted tuples.
        insert_df = pd.DataFrame(insert_tuples)[all_keys]
        frame = pd.concat([frame, insert_df], ignore_index=True)

    return frame, len(delete_tuples) > 0 or len(insert_tuples) > 0
