package edu.zjunb.llm.yeya.service;
import edu.zjunb.llm.yeya.service.FileProcessingService;
import edu.zjunb.llm.yeya.config.RabbitConfig;

import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class MessageConsumer {

    @Autowired
    private FileProcessingService fileProcessor;

    @RabbitListener(queues = RabbitConfig.FILE_PROCESSING_QUEUE)
    public void handleFileRequest(String filePath) {
        fileProcessor.processFileAsync(filePath);
    }
}