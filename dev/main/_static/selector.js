/* The version selector is intentionally dependency-free and relative-URL only. */
(function () {
  "use strict";

  function start() {
    var config = window.HTTK_DOCS_VERSIONING;
    if (!config || !config.versionPathDepth) return;
    var contentRoot = document.documentElement.dataset.content_root;
    if (!contentRoot) return;
    var rootUrl = new URL(contentRoot, document.baseURI);
    var siteRootUrl = new URL("../".repeat(config.versionPathDepth), rootUrl);
    var siteRoot = siteRootUrl.pathname;
    fetch(siteRoot + "versions.json").then(function (response) {
      if (!response.ok) throw new Error("versions unavailable");
      return response.json();
    }).then(function (manifest) {
      var brand = document.querySelector(".sidebar-brand");
      if (!brand || !manifest || !Array.isArray(manifest.versions)) return;
      var wrapper = document.createElement("div");
      wrapper.className = "httk-version-selector" + (config.channel === "dev" ? " httk-version-selector--dev" : "");
      var label = document.createElement("label");
      label.textContent = "Documentation version";
      label.setAttribute("for", "httk-documentation-version");
      var select = document.createElement("select");
      select.id = "httk-documentation-version";
      select.setAttribute("aria-label", "Documentation version");
      manifest.versions.forEach(function (item) {
        var option = document.createElement("option");
        option.value = item.path;
        option.textContent = item.channel === "dev" ? item.name + " (development)" : item.name;
        option.dataset.versionName = item.name;
        option.selected = item.name === config.version;
        select.appendChild(option);
      });
      select.addEventListener("change", function () {
        var target = select.options[select.selectedIndex];
        if (!target || !target.value) return;
        var targetRoot = new URL(target.value, siteRootUrl);
        var currentRoot = new URL(config.version === "dev:main" ? "dev/main/" : config.version + "/", siteRootUrl);
        var currentUrl = new URL(window.location.href);
        var currentPagePath = "";
        if (currentUrl.pathname.indexOf(currentRoot.pathname) === 0) {
          currentPagePath = currentUrl.pathname.slice(currentRoot.pathname.length);
        } else {
          currentRoot = new URL("latest/", siteRootUrl);
          if (currentUrl.pathname.indexOf(currentRoot.pathname) === 0) {
            currentPagePath = currentUrl.pathname.slice(currentRoot.pathname.length);
          }
        }
        fetch(siteRoot + target.value + "pages.json").then(function (response) {
          if (!response.ok) throw new Error("pages unavailable");
          return response.json();
        }).then(function (pages) {
          var page = currentPagePath || "index.html";
          var exists = pages && Array.isArray(pages.pages) && pages.pages.indexOf(page) !== -1;
          window.location.href = siteRoot + target.value + (exists ? page : "") + window.location.hash;
        }).catch(function () {
          window.location.href = targetRoot.pathname + window.location.hash;
        });
      });
      wrapper.appendChild(label);
      wrapper.appendChild(select);
      brand.parentNode.insertBefore(wrapper, brand.nextSibling);
    }).catch(function () { /* Local builds have no site manifest. */ });
  }
  document.addEventListener("DOMContentLoaded", start);
}());
