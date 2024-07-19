package edu.zjunb.llm.yeya.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

import java.nio.file.Path;

@Service
public class FileProcessingService {

    @Autowired
    private RestTemplate restTemplate;

    @Async
    public void processFileAsync(String filePath) {
        String url = "http://localhost:5000/process_file";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        String jsonRequest = "{\"file_path\":\"" + filePath + "\"}";
        System.out.println("filePath "+filePath);
        HttpEntity<String> request = new HttpEntity<>(jsonRequest, headers);

        String response = restTemplate.postForObject(
                url,
                request,
                String.class);

        System.out.println("Python Script Output: " + response);
    }
}