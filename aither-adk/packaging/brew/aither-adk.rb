# Homebrew formula for AitherADK
# Install: brew install aither-adk
# Or from tap: brew tap aitherium/tap && brew install aither-adk

class AitherAdk < Formula
  include Language::Python::Virtualenv

  desc "Agent Development Kit for AitherOS — build AI agent fleets with any LLM"
  homepage "https://aitherium.com"
  url "https://files.pythonhosted.org/packages/source/a/aither-adk/aither_adk-0.9.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "Proprietary"

  depends_on "python@3.12"

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/source/h/httpx/httpx-0.27.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/P/PyYAML/pyyaml-6.0.2.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/source/f/fastapi/fastapi-0.115.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/source/u/uvicorn/uvicorn-0.30.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  def install
    virtualenv_install_with_resources
  end

  def post_install
    # Create config directory
    (var/"aither").mkpath
  end

  test do
    assert_match "AitherADK", shell_output("#{bin}/aither --help")
  end
end
