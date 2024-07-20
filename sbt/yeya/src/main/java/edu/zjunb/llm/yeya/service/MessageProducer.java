package edu.zjunb.llm.yeya.service;

import edu.zjunb.llm.yeya.config.RabbitConfig;

import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MessageProducer {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void sendFileRequest(String filePath) {
        rabbitTemplate.convertAndSend(RabbitConfig.FILE_PROCESSING_QUEUE, filePath);
    }
}