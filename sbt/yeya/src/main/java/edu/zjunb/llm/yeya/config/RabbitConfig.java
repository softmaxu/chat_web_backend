package edu.zjunb.llm.yeya.config;

import org.springframework.amqp.core.Queue;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitConfig {

    public static final String FILE_PROCESSING_QUEUE = "fileProcessingQueue";

    @Bean
    public Queue fileProcessingQueue() {
        return new Queue(FILE_PROCESSING_QUEUE, true);
    }
}