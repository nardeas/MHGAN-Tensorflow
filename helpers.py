import tensorflow as tf

def create_summary_helper(sess, output_path):
    
    with tf.name_scope('generator'):
        generator_loss_history = tf.placeholder(
            tf.float32,
            [ None ],
            name='loss_history_placeholder'
        )
        generator_mean_loss = tf.reduce_mean(
            generator_loss_history,
            name='mean_loss_placeholder'
        )
        generator_summary = tf.summary.merge([
            tf.summary.scalar('loss', generator_loss_history[-1]),
            tf.summary.scalar('mean_loss', generator_mean_loss),
            tf.summary.histogram('loss_history', generator_loss_history)
        ])
        
    with tf.name_scope('discriminator'):
        discriminator_loss_history = tf.placeholder(
            tf.float32,
            [ None ],
            name='loss_history_placeholder'
        )
        discriminator_mean_loss = tf.reduce_mean(
            discriminator_loss_history,
            name='mean_loss_placeholder'
        )
        discriminator_summary = tf.summary.merge([
            tf.summary.scalar('loss', discriminator_loss_history[-1]),
            tf.summary.scalar('mean_loss', discriminator_mean_loss),
            tf.summary.histogram('loss_history', discriminator_loss_history)
        ])
        
    g_writer = tf.summary.FileWriter(
        output_path + '/generator',
        sess.graph
    )
    d_writer = tf.summary.FileWriter(
        output_path + '/discriminator',
        #sess.graph
    )
   
    def add_summaries(epoch, accumulate_losses):
        g_writer.add_summary(sess.run(
            generator_summary,
            feed_dict={
                generator_loss_history: accumulate_losses.T[0]
            }),
            epoch
        )
        d_writer.add_summary(sess.run(
            discriminator_summary,
            feed_dict={
                discriminator_loss_history: accumulate_losses.T[1]
            }),
            epoch
        )
        
    return add_summaries
            