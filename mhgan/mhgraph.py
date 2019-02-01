import tensorflow as tf

'''
Returns an in-graph Metropolis-Hastings sampler for WGAN.

NOTE: there seems to be significant overhead from nested looping,
so don't use this. It's for reference only until a better solution is found.
'''
def create_metropolis_hastings_sampler(
    fake_output_tensor,
    real_output_tensor,
    generator_output_shape,
    parallel_ratios=(15, 15)):

    # Placeholders to return c samples while sampling k times each round
    c = tf.placeholder(tf.int32, [], name='round_count')
    k = tf.placeholder(tf.int32, [], name='k_samples_per_round')

    # Uniform noise for computing sample acceptance
    u = tf.reshape(tf.random_uniform([ c, k ]), [-1])

    # Selected samples will be concatenated to this in every major sampling round
    selected = tf.constant([], shape=[0, *generator_output_shape])

    # Flat scores for real + fake samples
    scores = tf.reshape(
        # Re-group
        tf.reshape(
            # Add calibration scores from real discriminator output
            tf.concat([
                # Convert WGAN outputs to sigmoid
                tf.sigmoid(real_output_tensor),
                tf.sigmoid(fake_output_tensor)
            ], 0),
            (c, k + 1)
        ),
        [-1]
    )

    # Currently accepted sample id
    idx_selected = tf.constant(0)
    # Currently active MCMC round
    idx_round = tf.constant(0)
    # Loop id
    idx = tf.constant(0)

    samples = tf.while_loop(
        # Draw a total of c MH samples
        lambda idx_round, _: tf.less(idx_round, c),
        # For each round, concatenated the picked sample
        lambda idx_round, selected: (
            tf.add(idx_round, 1),
            tf.concat([
                selected,
                tf.expand_dims(
                    # Select from generator output by id returned by inner loop
                    gan.G[
                        # Subtract current round * k since ids are rolling and scores are
                        # flat
                        tf.subtract(
                            tf.while_loop(
                            # Like above, make idx relative to round
                            lambda idx, _: tf.less(tf.subtract(idx, idx_round * k), k),
                            lambda idx, idx_selected: (
                                tf.add(idx, 1),
                                # Metropolis-Hastings part
                                tf.cond(
                                    # If u > acceptance, sample index is rejected
                                    # If u <= acceptance, sample index is accepted and idx_selected
                                    # corresponds to the current best sample
                                    tf.greater(u[idx] - tf.minimum(1.,
                                        # Compute acceptance rate, note that the value of idx lags behind
                                        # as we want to compare idx_selected to the next value. During
                                        # first round idx_selected points to the calibration sample
                                        tf.div(
                                            tf.subtract(tf.div(1., scores[idx_selected]), 1),
                                            tf.subtract(tf.div(1., scores[idx + 1]), 1)
                                        )
                                    ), 0),
                                    # Reject
                                    lambda: idx_selected,
                                    # Accept
                                    lambda: idx
                                )
                            ),
                            loop_vars=[
                                idx,
                                idx_selected
                            ],
                            parallel_iterations=parallel_ratios[1],
                            back_prop=False,
                            swap_memory=False
                        # 0 => idx, 1 => idx_selected
                        )[1],
                    idx_round * k)
                    ],
                    0
                )],
                # Concat newly accepted best sample on main axis
                0
            ),
        ),
        loop_vars=[
            idx_round,
            selected
        ],
        # Shape invariants are provided since each round the array of selected
        # samples grows
        shape_invariants=[
            idx_round.get_shape(),
            tf.TensorShape([ None, *generator_output_shape ])
        ],
        parallel_iterations=parallel_ratios[0],
        back_prop=False,
        swap_memory=False
    # 0 => idx_round, 1 => selected samples array
    )[1]

    # Higher-order function for generating enhanced samples
    def sampler(
        sess,
        noise_sampler,
        calibration_sampler,
        generator_input_tensor,
        generator_input_shape,
        discriminator_input_tensor,
        total_count=1,
        k_count=100,
        ):
        return sess.run(samples, feed_dict={
            # Candidates
            generator_input_tensor: noise_sampler(shape=[
                total_count * k_count,
                generator_input_shape
            ]),
            # Calibration
            discriminator_input_tensor: calibration_sampler(total_count),
            # Number of images generated
            c: total_count,
            # Number of samples per round
            k: k_count
        })

    return sampler
