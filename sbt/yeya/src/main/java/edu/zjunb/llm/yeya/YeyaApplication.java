package edu.zjunb.llm.yeya;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
public class YeyaApplication {

	public static void main(String[] args) {
		SpringApplication.run(YeyaApplication.class, args);
        System.out.println("Hello, World!");
	}

	@RestController
    public static class HelloWorldController {

        @GetMapping("/")
        public String helloWorld() {
            return "Hello, World!";
        }
    }
}
