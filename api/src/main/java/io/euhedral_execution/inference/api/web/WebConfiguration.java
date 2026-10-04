package io.euhedral_execution.inference.api.web;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/// Watches the connections of generation requests ([ServletClientLink]).
@Configuration(proxyBeanMethods = false)
public class WebConfiguration implements WebMvcConfigurer {
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new ServletClientLink.Interceptor()).addPathPatterns("/v1/**");
    }
}
