FROM sipgisr/opencv-grpc:generic-latest

COPY PaperTablet.py /workspace/external.py

COPY Template.png /workspace/Template.png

EXPOSE 8061

CMD ["python", "service.py"]
