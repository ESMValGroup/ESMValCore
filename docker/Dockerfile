# Recipe from: https://tech.quantco.com/blog/pixi-production
# To build this container, go to ESMValCore root folder and execute:
# docker build --pull -t esmvalcore:latest . -f docker/Dockerfile
FROM ghcr.io/prefix-dev/pixi:latest AS build

WORKDIR /app
COPY . .
RUN pixi global install git
RUN pixi install --frozen -e esmvalcore
RUN pixi shell-hook -e esmvalcore -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'esmvaltool "$@"' >> /app/entrypoint.sh

FROM ubuntu:latest AS production
WORKDIR /app
# Copy the environment:
COPY --from=build /app/.pixi/envs/esmvalcore /app/.pixi/envs/esmvalcore
# Copy the source code because esmvalcore is installed in editable mode:
COPY --from=build /app/esmvalcore /app/esmvalcore
# Copy the entrypoint script and make it executable:
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
